class AnalysisImplementation:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        ret = type(self).__name__
        if len(self.kwargs) > 0:
            ret = ret + ": " + str(self.kwargs)  # TODO: args?
        return ret

    def run(self, conn):
        return self._run(conn, *self.args, **self.kwargs)

    @staticmethod
    def _run(*args, **kwargs):
        raise NotImplementedError()
