from multiprocessing import Process as _Process

import dill


class Process(_Process):
    def __init__(self, *args, kwargs=None, **kwds):
        # note that the kwargs are not passed to the parent class
        super().__init__(*args, **kwds)
        # use dill to serialize kwargs instead
        self._kwargs = dill.dumps(kwargs)

    def run(self):
        if self._target:
            # deserialize the kwargs before executing
            self._kwargs = dill.loads(self._kwargs)
            self._target(*self._args, **self._kwargs)
