import torch 

class BaseAlgo:
    def __init__(self, args, model, tokenizer, device):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def prepare_batch(self, steps):
        """
        Turn a list[Step] into tensors on self.dev.
        Return whatever update() needs.
        """
        raise NotImplementedError

    def update(self, batch):
        """
        One gradient update on *this worker only*.
        Must call .backward() but NOT .step().
        Return latest loss as float (for logging).
        """
        raise NotImplementedError