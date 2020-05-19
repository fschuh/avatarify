from arguments import opt
from networking import SerializingContext
from utils import log, TicToc, AccumDict, Once

import cv2
import numpy as np
import zmq
import msgpack
import msgpack_numpy as m
import threading
from predictor_worker import ThreadedBase, MessageBuffer, create_message
m.patch()


DEFAULT_PORT = 5556


class MessageReceiver(ThreadedBase):
    def __init__(self, message_buffer, predict_buffer, socket):
        super().__init__("MessageReceiver")
        self._predict_buffer = predict_buffer  # for result of predict_async messages
        self._message_buffer = message_buffer  # for the other messages
        self._socket = socket
        self._event = threading.Event()

    def run(self):
        while self._running:
            try:
                attr_recv, data_recv = self._socket.recv_data()
                if attr_recv == 'predict':
                    result = cv2.imdecode(np.frombuffer(data_recv, dtype='uint8'), -1)
                    self._predict_buffer.add(result)

                    # TODO: remove line!
                    #self._message_buffer.add(create_message(attr_recv, data_recv))
                else:
                    self._message_buffer.add(create_message(attr_recv, data_recv))
            except Exception as e:
                log("Exception: " + str(e))


class PredictorRemote:
    def __init__(self, *args, worker_host='localhost', worker_port=DEFAULT_PORT, predict_buffer, **kwargs):
        self.worker_host = worker_host
        self.worker_port = worker_port
        self.predictor_args = (args, kwargs)
        self.timing = AccumDict()

        self.address = f"tcp://{worker_host}:{worker_port}"
        self.context = SerializingContext()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect(self.address)

        self.message_buffer = MessageBuffer(name="RegularMessages")
        self.message_receiver = MessageReceiver(self.message_buffer, predict_buffer, self.socket)
        self.message_receiver.start()

        if not self.check_connection():
            self.socket.disconnect(self.address)
            # TODO: this hangs, as well as context.__del__
            # self.context.destroy()
            raise ConnectionError(f"Could not connect to {worker_host}:{worker_port}")

        log(f"Connected to {self.address}")

        self.init_worker()

    def check_connection(self, timeout=1000):
        msg = (
            'hello',
            [], {}
        )

        try:
            old_rcvtimeo = self.socket.RCVTIMEO
            self.socket.RCVTIMEO = timeout
            response = self._send_recv_msg(msg)
            self.socket.RCVTIMEO = old_rcvtimeo
        except zmq.error.Again:
            return False

        return response == 'OK'

    def init_worker(self):
        msg = (
            '__init__',
            *self.predictor_args,
        )
        return self._send_recv_msg(msg)

    def __getattr__(self, attr):
        return lambda *args, **kwargs: self._send_recv_msg((attr, args, kwargs))

    def _send_recv_msg(self, msg):
        attr, args, kwargs = msg

        tt = TicToc()
        tt.tic()
        if attr == 'predict' or attr == 'predict_async':
            image = args[0]
            assert isinstance(image, np.ndarray), 'Expected image'
            ret_code, data = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), opt.jpg_quality])
        else:
            data = msgpack.packb((args, kwargs))
        self.timing.add('PACK', tt.toc())

        tt.tic()
        if attr == 'predict_async':
            # host doesn't understand 'predict_async' message, so we send it a regular 'predict' instead
            self.socket.send_data('predict', data)
        else:
            self.socket.send_data(attr, data)
        self.timing.add('SEND', tt.toc())

        if attr == 'predict_async':
            # for async predicts, we don't want to wait for any messages to arrive so we return immediately
            return

        tt.tic()
        #attr_recv, data_recv = self.socket.recv_data()
        message = self.message_buffer.pop_next()
        attr_recv = message["attr"]
        data_recv = message["data"]
        self.timing.add('RECV', tt.toc())

        tt.tic()
        if attr_recv == 'predict':
            result = cv2.imdecode(np.frombuffer(data_recv, dtype='uint8'), -1)
        else:
            result = msgpack.unpackb(data_recv)
        self.timing.add('UNPACK', tt.toc())

        Once(self.timing, per=1)

        return result
