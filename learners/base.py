import torch 

class BaseAlgo:
    def __init__(self, cfg, model, tokenizer, device):
        self.cfg = cfg
        self.model = model
        self.tok = tokenizer
        self.dev = device
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

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