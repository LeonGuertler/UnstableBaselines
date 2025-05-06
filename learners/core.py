from trajectory_buffer import Step


class BaseLearner:
    def __init__(self, args):
        self.args = args 
        torch.cuda.set_device(0)
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.update_step = 0

        # gradient checkpointing
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # bfloat16 training
        if self.args.bf16_training:
            self.model.to(torch.bfloat16)

    def update(self, steps: List[Step]):
        raise NotImplementedError

        def store_model(self, checkpoint_folder: Optional[str], checkpoint_filename: Optional[str]):
        # if not provided, use defaults
        if checkpoint_folder is None:
            checkpoint_folder = self.args.output_dir_checkpoints
        if checkpoint_filename is None:
            checkpoint_filename = f"Update_Step_{self.update_step}"

        save_path = os.path.join(checkpoint_folder, checkpoint_filename)
        checkpoint = {"model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(), "args": vars(self.args)}
        torch.save(checkpoint, save_path)
        print(f"[REINFORCE] Checkpoint saved to {save_path}")

    def load_model(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[REINFORCE] Checkpoint loaded from {checkpoint_path}")

    def update_weights(self, weights: dict):
        with torch.no_grad():
            device = self.model.device
            state_dict = self.model.state_dict()
            for k in weights:
                if k in state_dict and state_dict[k].shape == weights[k].shape:
                    tensor = torch.from_numpy(weights[k].copy()).to(device)
                    state_dict[k].copy_(tensor)