import threading


class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)


def data_uploader():
    pass


def set_initializer():
    pass


def redirect_to_main():
    pass


thread1 = ThreadWithResult(target=data_uploader)
thread2 = ThreadWithResult(target=set_initializer)
thread3 = ThreadWithResult(target=redirect_to_main)

thread1.start()
thread2.start()
thread3.start()

thread1.join()
thread2.join()
thread3.join()
