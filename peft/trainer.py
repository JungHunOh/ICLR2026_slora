from transformers import Trainer
import torch

class SignPreservingLoRATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._opt_step_count = 0

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
            
        param_groups = self.optimizer.param_groups
        assert len(param_groups) > 0

        group0 = param_groups[0]

        lr = group0.get('lr', 1e-3)
        betas = group0.get('betas', (0.9, 0.999))
        eps = group0.get('eps', 1e-8)
        weight_decay = group0.get('weight_decay', 0.0)
        amsgrad = getattr(self.optimizer, 'amsgrad', False)
        params = group0['params']

        self.optimizer = SignPreservingAdamW(
            params,
            model=self.model,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )

        optimizer = self.optimizer
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)
    

class SignPreservingAdamW(torch.optim.AdamW):
    def __init__(self, params, model=None, **kwargs):
        super().__init__(params, **kwargs)
        self.model = model
        self._step_count = 0

    def step(self, closure=None):
        loss = super().step(closure)
        self._step_count += 1
        if self._step_count % 5 == 0:
            self.sign_preserve_fn(self.model)
        return loss
    
    def sign_preserve_fn(self, model):
        with torch.no_grad():
            i = 0
            for module in self.model.modules():
                if hasattr(module, 'base_layer') and hasattr(module, 'lora_A'):
                    W = module.base_layer.weight  # base weight
                    A = module.lora_A['default'].weight
                    B = module.lora_B['default'].weight
                    scaling = module.scaling['default']

                    # LoRA effective update: ΔW = B @ A
                    delta = B @ A * scaling  # [out_features, in_features]
                    W_eff = W + delta.to(W.dtype)

                    # Update W: preserve original sign, adopt W_eff's magnitude
                    same_sign = (module.initial_sign == (W_eff >= 0))
                    
                    if i == 10:
                        print(same_sign.float().mean())
                    i += 1

                    W_new = torch.where(same_sign, W, W + torch.sign(-W_eff) * torch.abs(W_eff))
                    module.base_layer.weight.data = W_new
