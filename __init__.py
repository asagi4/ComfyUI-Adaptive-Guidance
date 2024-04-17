import comfy.samplers
import torch

cos = torch.nn.CosineSimilarity(dim=1)


class AdaptiveGuider(comfy.samplers.CFGGuider):
    threshold_timestep = 0

    def set_threshold(self, threshold):
        self.threshold = threshold

    def check_cos_sim(self, ts, cond_pred, uncond_pred):
        # Is this reshape correct? It at least gives a scalar value...
        sim = cos(cond_pred.reshape(1, -1), uncond_pred.reshape(1, -1)).item()
        sim = round(sim, 4)
        if sim > self.threshold:
            print("AdaptiveGuidance: Cosine similarity", sim, "exceeds threshold, setting CFG to 1.0")
            self.threshold_timestep = ts

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        cond = self.conds.get("positive")
        uncond = self.conds.get("negative")
        ts = timestep[0].item()
        if self.threshold_timestep > ts:
            return comfy.samplers.sampling_function(
                self.inner_model, x, timestep, uncond, cond, 1.0, model_options=model_options, seed=seed
            )
        else:
            self.threshold_timestep = 0
            uncond_pred, cond_pred = comfy.samplers.calc_cond_batch(
                self.inner_model, [uncond, cond], x, timestep, model_options
            )
            self.check_cos_sim()
            return comfy.samplers.cfg_function(
                self.inner_model,
                cond_pred,
                uncond_pred,
                self.cfg,
                x,
                timestep,
                model_options=model_options,
                cond=cond,
                uncond=uncond,
            )


class AdaptiveGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "threshold": ("FLOAT", {"default": 0.990, "min": 0.90, "max": 1.0, "step": 0.001, "round": 0.001}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "patch"

    CATEGORY = "sampling/custom_sampling/guiders"

    def patch(self, model, positive, negative, threshold, cfg):
        g = AdaptiveGuider(model)
        g.set_conds(positive, negative)
        g.set_cfg(cfg)
        g.set_threshold(threshold)

        return (g,)


class LinearAdaptiveGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "threshold": ("FLOAT", {"default": 0.990, "min": 0.90, "max": 1.0, "step": 0.001, "round": 0.001}),
                "betas_cond": ("STRING", {"default": "0.4,0.2,0.05"}),
                "betas_uncond": ("STRING", {"default": "0.4,0.2,0.05"}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "patch"

    CATEGORY = "sampling/custom_sampling/guiders"

    def patch(self, model, positive, negative, cfg, threshold, betas_cond, betas_uncond):
        g = LinearAdaptiveGuider(model)
        g.set_conds(positive, negative)
        g.set_cfg(cfg)
        g.set_threshold(threshold)

        def split_floats(string):
            return [float(x.strip()) for x in string.split(",")]

        g.set_betas(split_floats(betas_cond), split_floats(betas_uncond))

        return (g,)


class LinearAdaptiveGuider(AdaptiveGuider):
    last_seen_sigma = 0

    def set_betas(self, beta_cond, beta_uncond):
        self.beta_cond = beta_cond
        self.beta_uncond = beta_uncond

    def get_beta(self, beta_list):
        idx = min(self.counter - 1, len(beta_list) - 1)
        return beta_list[idx]

    def initialize(self):
        self.cond_results = []
        self.uncond_results = []
        self.counter = 0

    def predict_linear(self):
        return torch.stack(self.cond_results, dim=0).sum(dim=0) + torch.stack(self.uncond_results, dim=0).sum(dim=0)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        cond = self.conds.get("positive")
        uncond = self.conds.get("negative")
        ts = timestep[0].item()
        # Not exactly correct, but will work
        if self.last_seen_sigma < ts:
            self.initialize()
        self.last_seen_sigma = ts
        self.counter += 1
        if ts < self.threshold_timestep:
            return comfy.samplers.sampling_function(
                self.inner_model, x, timestep, uncond, cond, 1.0, model_options=model_options, seed=seed
            )

        else:
            self.threshold_timestep = 0
        bc = self.get_beta(self.beta_cond)
        buc = self.get_beta(self.beta_uncond)
        print(f"LinearAdaptive: {bc=} {buc=}")
        if self.counter % 2 != 0:
            # cfg step
            print("LinearAdaptive: Full CFG step")
            uncond_pred, cond_pred = comfy.samplers.calc_cond_batch(
                self.inner_model, [uncond, cond], x, timestep, model_options
            )
            self.cond_results.append(cond_pred * bc)
            self.uncond_results.append(uncond_pred * buc)
        else:
            # non-cfg step
            print("LinearAdaptive: Estimated CFG step")
            cond_pred = comfy.samplers.calc_cond_batch(self.inner_model, [cond], x, timestep, model_options)[0]
            self.cond_results.append(cond_pred * bc)
            uncond_pred = self.predict_linear()
            self.uncond_results.append(uncond_pred * buc)

        self.check_cos_sim(ts, cond_pred, uncond_pred)
        return comfy.samplers.cfg_function(
            self.inner_model,
            uncond_pred,
            cond_pred,
            self.cfg,
            x,
            timestep,
            model_options=model_options,
            cond=cond,
            uncond=uncond,
        )


NODE_CLASS_MAPPINGS = {
    "AdaptiveGuidance": AdaptiveGuidance,
    "LinearAdaptiveGuidance": LinearAdaptiveGuidance,
}
