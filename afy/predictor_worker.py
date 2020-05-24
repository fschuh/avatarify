from predictor_local import PredictorLocal
from arguments import opt
from networking import SerializingContext
from utils import log, TicToc, AccumDict, Once

import cv2
import numpy as np
import zmq
import msgpack
import msgpack_numpy as m
import threading
m.patch()


def create_message(attr, data):
    log("Creating message for sending: {}".format(attr))
    return {
        "attr": attr,
        "data": data
    }


class MessageBuffer(object):
    MAX_SIZE = 1024 * 32

    def __init__(self, on_update_callback=None, lock=None, name="MessageBuffer"):
        self._buf = []

        if lock is None:
            self._lock = threading.Lock()
        else:
            self._lock = lock

        self._event = threading.Event()
        self._on_update_callback = on_update_callback
        self._name = name
        self._debug = True

    @property
    def size(self):
        return len(self._buf)

    def clear(self):
        with self._lock:
            self._buf.clear()
    
    def add(self, message):
        with self._lock:
            if len(self._buf) >= MessageBuffer.MAX_SIZE:
                # buffer discard message
                log("Message buffer size limit reached, discarding message")
                return

            prev_size = self.size
            self._buf.append(message)
            self._debug_log("MessageBuffer '{}': message added, size: {}".format(self._name, self.size))
            self._event.set()
            if self._on_update_callback:
                self._on_update_callback(self, prev_size, self.size)
    
    def pop_next(self):
        self._event.wait()
        return self._pop_next()

    def pop_next_non_blocking(self):
        return self._pop_next()

    def _pop_next(self):
        with self._lock:
            if len(self._buf) == 0:
                return None

            prev_size = self.size
            item = self._buf.pop(0)
            self._debug_log("MessageBuffer '{}': message removed, size: {}".format(self._name, self.size))
            if self._on_update_callback:
                self._on_update_callback(self, prev_size, self.size)

            if self.size == 0:
                self._event.clear()

            return item

    def _debug_log(self, message):
        if self._debug:
            log(message)



class ThreadedBase(object):
    def __init__(self, thread_name):
        self._running = False
        self._thread_name = thread_name
        self._thread = threading.Thread(target=self.run, name=thread_name, daemon=None)
    
    def start(self):
        if self._running:
            log("{} already started! Please stop first. Ignoring.".format(self._thread_name))
            return

        self._running = True
        log("{} thread started".format(self._thread_name))
        self._thread.start()

    def stop(self):
        self._running = False
        log("{} thread stopped".format(self._thread_name))

    def is_running(self):
        return self._running

    def run(self):
        pass


class MessageSender(ThreadedBase):
    def __init__(self, send_buffer, socket):
        super().__init__("MessageSender")
        self._send_buffer = send_buffer
        self._socket = socket
        self._event = threading.Event()

    def run(self):
        while self._running:
            message = self._send_buffer.pop_next()
            if message is not None:
                #log("sending message")
                self._socket.send_data(message["attr"], message["data"])


# class PredictorWorker(ThreadedBase):
#     def __init__(self, buffer):
#         super().__init__("PredictorWorker")
#         self._buffer = buffer
    
#     def run(self, predictor):
#         if self.is_running():
#             log("PredictorWorker already started! Please stop first. Ignoring.")
#             return

#         self._predictor = predictor
#         self._running = True
#         t = threading.Thread(target=self._thread_run)
#         log("PredictorWorker thread started")
#         t.start()        

#     def run(self):
#         while self._running:
#             message = self._buffer.pop_next()
#             if message is not None:
#                 attr = message["attr"]
#                 data = message["data"]
#                 self._predictor.predict(message)


def message_handler(socket, send_buffer):
    predictor = None
    predictor_args = ()
    timing = AccumDict()
    
    try:
        while True:
            tt = TicToc()

            tt.tic()
            attr, data = socket.recv_data()
            timing.add('RECV', tt.toc())

            try:
                tt.tic()
                if attr == 'predict':
                    image = cv2.imdecode(np.frombuffer(data, dtype='uint8'), -1)
                else:
                    args = msgpack.unpackb(data)
                timing.add('UNPACK', tt.toc())
            except ValueError:
                log("Invalid Message")
                continue

            tt.tic()
            if attr == "hello":
                result = "OK"
            elif attr == "__init__":
                if args == predictor_args:
                    log("Same config as before... reusing previous predictor")
                else:
                    del predictor
                    predictor_args = args
                    predictor = PredictorLocal(*predictor_args[0], **predictor_args[1])
                    log("Initialized predictor with:", predictor_args)
                result = True
                tt.tic() # don't account for init
            elif attr == 'predict':
                result = getattr(predictor, attr)(image)
            else:
                result = getattr(predictor, attr)(*args[0], **args[1])
            timing.add('CALL', tt.toc())

            tt.tic()
            if attr == 'predict':
                assert isinstance(result, np.ndarray), 'Expected image'
                ret_code, data_send = cv2.imencode(".jpg", result, [int(cv2.IMWRITE_JPEG_QUALITY), opt.jpg_quality])
            else:
                data_send = msgpack.packb(result)
            timing.add('PACK', tt.toc())

            tt.tic()
            #socket.send_data(attr, data_send)
            #log("adding message to send queue")
            send_buffer.add(create_message(attr, data_send))
            timing.add('SEND', tt.toc())

            Once(timing, per=1)
    except KeyboardInterrupt:
        pass

def run_worker(port):
    log("Creating socket")
    context = SerializingContext()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:%s" % port)
    log("Listening for messages on port:", port)

    send_buffer = MessageBuffer()
    recv_buffer = MessageBuffer()

    #worker = PredictorWorker(message_buffer)
    message_sender = MessageSender(send_buffer, socket)
    message_sender.start()
    message_handler(socket, send_buffer)
    message_sender.stop()
