class AttrDict(dict):
    def __init__(self, seq=None, **kwargs):
        if seq is None:
            seq = {}
        super(self.__class__, self).__init__(seq, **kwargs)
        self.__dict__ = self
