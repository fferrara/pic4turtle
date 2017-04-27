"""
RPC Server
"""

from jsonrpctcp import server

class CommandServer(server.Server):
    def __init__(self, host, port, photo_handler):
        server.Server.__init__(self, (host, port))
        self.photo_handler = photo_handler

        # add each remote procedure handler
        self.add_handler(self.classify_photo, 'classify')

    def classify_photo(self, image):
        try:
            # classify image
            result = self.photo_handler.recognize(image).get_first_class()

            return result[0]  # just the class, without confidence
        except ValueError as ve:  # image not valid
            raise Exception('Image %s not valid: %s' % (image, ve.message))
        except Exception as e:  # generic error
            print e.message
            raise Exception('Generic error classifying %s' % image)

if __name__ == '__main__':
    import threading
    import time
    import sys
    import handler
    import config

    h = handler.PhotoHandler()
    svr = CommandServer(config.conf.RpcHost, int(config.conf.RpcPort), h)

    server_thread = threading.Thread(target=svr.serve)
    server_thread.daemon = True
    server_thread.start()

    try:
        print 'Server started'
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print 'Finished.'
        sys.exit()