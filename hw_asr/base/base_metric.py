class BaseMetric:
    def __init__(
            self, name=None, beam_size=100, ctc_type='base', *args, **kwargs):
        self.name = name if name is not None else type(self).__name__
        self.beam_size = beam_size
        self.ctc_type = ctc_type

    def __call__(self, **batch):
        raise NotImplementedError()
