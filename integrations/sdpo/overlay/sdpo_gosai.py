import torch

from diffusion_gosai_update import Diffusion, Loss


class DiffusionSDPO(Diffusion):
    def __init__(self, config, ref_model=None, beta=1.0, eval=False, generator="rkl"):
        super().__init__(config, eval=eval)
        self.beta = beta
        self.ref_model = ref_model
        self.generator = generator
        if ref_model is not None:
            self.set_ref_model(ref_model)

    def set_ref_model(self, ref_model):
        self.ref_model = ref_model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _forward_pass_diffusion_t(self, x0, t):
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            t += 1 / self.T
        if self.change_of_variables:
            unet_conditioning = t[:, None]
            f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
            f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))[:, None]
        else:
            sigma, dsigma = self.noise(t)
            unet_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])

        xt = self._ordered_training_xt(
            x0,
            move_chance,
            unet_conditioning,
            rewards=getattr(self, "_ordering_rewards", None),
        )
        model_output = self.forward(xt, unet_conditioning)
        log_p_theta = torch.gather(
            input=model_output,
            dim=-1,
            index=x0[:, :, None],
        ).squeeze(-1)

        if self.change_of_variables or self.importance_sampling:
            return log_p_theta * torch.log1p(-torch.exp(-self.noise.sigma_min))
        return -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]

    def _loss_t(self, x0, t, attention_mask):
        input_tokens, output_tokens, attention_mask = self._maybe_sub_sample(
            x0, attention_mask
        )
        if self.parameterization == "ar":
            logprobs = self.backbone(input_tokens, None)
            loss = -logprobs.gather(-1, output_tokens[:, :, None])[:, :, 0]
        else:
            loss = self._forward_pass_diffusion_t(input_tokens, t)

        nlls = loss * attention_mask
        token_nll = nlls.sum(dim=1) / attention_mask.sum(dim=1)
        return Loss(loss=token_nll, nlls=nlls, token_mask=attention_mask)

    def _compute_loss_t(self, batch, t, prefix):
        losses = self._loss_t(
            batch["seqs"],
            t,
            batch.get("attention_mask"),
        )
        if prefix == "train":
            self.train_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.train_metrics
        elif prefix == "val":
            self.valid_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.valid_metrics
        elif prefix == "test":
            self.test_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.test_metrics
        else:
            raise ValueError(f"Invalid prefix: {prefix}")
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        return losses.loss

    def training_step(self, batch, batch_idx):
        del batch_idx
        x0 = batch["seqs"]
        # This is the sample-level HepG2 utility used by the upstream SDPO loss.
        rewards = batch["clss"][:, 0]
        t = self._sample_t(x0.shape[0], x0.device)

        self._ordering_rewards = rewards
        if self.ref_model is not None:
            self.ref_model._ordering_rewards = rewards
        self._dprm_stats_from_ordered_train = False
        model_log_prob = self._compute_loss_t(
            {"seqs": x0, "attention_mask": batch["attention_mask"]},
            t,
            "train",
        )
        ref_log_prob = self.ref_model._compute_loss_t(
            {"seqs": x0, "attention_mask": batch["attention_mask"]},
            t,
            "train",
        )
        if not getattr(self, "_dprm_stats_from_ordered_train", False):
            self.update_dprm_stats_from_batch(x0, rewards, t)
        self._ordering_rewards = None
        if self.ref_model is not None:
            self.ref_model._ordering_rewards = None

        g_theta = -self.beta * (model_log_prob - ref_log_prob)
        if self.generator == "rkl":
            lsm = torch.log_softmax(rewards, dim=0) - torch.log_softmax(
                g_theta, dim=0
            )
            return (torch.softmax(rewards, dim=0) * lsm).sum()
        if self.generator == "kl":
            lsm = torch.log_softmax(g_theta, dim=0) - torch.log_softmax(
                rewards, dim=0
            )
            return (torch.softmax(g_theta, dim=0) * lsm).sum()
        raise ValueError(f"Unsupported SDPO generator: {self.generator}")
