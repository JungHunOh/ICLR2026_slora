from transformers import Trainer
import torch
from peft.tuners.lora.layer import LoraLayer

class SignPreservingLoRATrainer(Trainer):
    def __init__(self, reg_lambda, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._opt_step_count = 0
        self.reg_lambda = reg_lambda

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
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        reg_loss = 0
        num = 0
        for module in model.modules():
            if isinstance(module, LoraLayer):
                B = module.lora_B['default'].weight
                A = module.lora_A['default'].weight

                ## alignment
                norm_A = torch.norm(A, p='fro')
                norm_B = torch.norm(B, p='fro')
                reg_loss += norm_A + norm_B
                num += 1
                # if norm_A * norm_B > 0:
                #     norm_BA = torch.norm(B@A, p='fro')
                #     reg_loss += norm_A * norm_B / norm_BA
                #     num += 1

                ## orthonormality
                #BtB = B.T @ B
                #AAt = A @ A.T
                #reg_loss += torch.norm(BtB - torch.eye(BtB.size(0),device=B.device), p='fro') ** 2 + torch.norm(AAt - torch.eye(AAt.size(0),device=A.device), p='fro') ** 2
                #num += 1
        if num > 0:
            reg_loss = reg_loss / num

        print(reg_loss)
        
        if return_outputs:
            (loss, outputs) = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        else:
            loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        
        loss = loss + reg_loss * self.reg_lambda

        return (loss, outputs) if return_outputs else loss

class SignPreservingAdamW(torch.optim.AdamW):
    def __init__(self, params, model=None, **kwargs):
        super().__init__(params, **kwargs)
        self.model = model
        self._step_count = 0

    def step(self, closure=None):
        loss = super().step(closure)
        self._step_count += 1
        # if self._step_count % 1 == 0:
        #     self.sign_preserve_fn(self.model)
        return loss
    
    def sign_preserve_fn(self, model):
        with torch.no_grad():
            i = 0
            for module in self.model.modules():
                #if hasattr(module, 'base_layer') and hasattr(module, 'lora_A'):
                if isinstance(module, LoraLayer):
                    W = module.base_layer.weight  # base weight
                    A = module.lora_A['default'].weight
                    B = module.lora_B['default'].weight
                    scaling = module.scaling['default']

                    # LoRA effective update: ΔW = B @ A
                    delta = B @ A * scaling  # [out_features, in_features]
                    W_eff = W + delta.to(W.dtype)   

                    # Update W: preserve original sign, adopt W_eff's magnitude
                    #same_sign = (module.initial_sign == (W_eff >= 0))
                    same_sign = (torch.sign(W) == torch.sign(W_eff))

                    if i % 7 == 0:
                        print(same_sign.float().mean())
                    i += 1

                    W_new = torch.where(same_sign, W, W - W_eff)
                    module.base_layer.weight.data = W_new
