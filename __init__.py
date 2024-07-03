import comfy.samplers
import comfy_extras.nodes_perpneg
import torch

cos = torch.nn.CosineSimilarity(dim=1)


class AdaptiveGuider(comfy.samplers.CFGGuider):
    threshold_timestep = 0
    uz_scale = 0.0

    def set_cfg(self, cfg):
        self.cfg = cfg

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_uncond_zero_scale(self, scale):
        self.uz_scale = scale

    def zero_cond(self, args):
        cond = args["cond_denoised"]
        x = args["input"]
        x -= x.mean()
        cond -= cond.mean()
        return x - (cond / cond.std() ** 0.5) * self.uz_scale

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        cond = self.conds.get("positive")
        uncond = self.conds.get("negative")
        ts = timestep[0].item()
        if self.threshold_timestep > ts:
            if self.uz_scale > 0.0:
                model_options = model_options.copy()
                model_options["sampler_cfg_function"] = self.zero_cond
            return comfy.samplers.sampling_function(
                self.inner_model, x, timestep, uncond, cond, 1.0, model_options=model_options, seed=seed
            )
        self.threshold_timestep = 0
        uncond_pred, cond_pred = comfy.samplers.calc_cond_batch(
            self.inner_model, [uncond, cond], x, timestep, model_options
        )
        if not self.threshold >= 1.0:
            # Is this reshape correct? It at least gives a scalar value...
            sim = cos(cond_pred.reshape(1, -1), uncond_pred.reshape(1, -1)).item()
            if sim >= self.threshold:
                print("AdaptiveGuidance: Cosine similarity", sim, "exceeds threshold, setting CFG to 1.0")
                self.threshold_timestep = ts
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
            },
            "optional": {"uncond_zero_scale": ("FLOAT", {"default": 0.0, "max": 2.0, "step": 0.01})},
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "patch"

    CATEGORY = "sampling/custom_sampling/guiders"

    def patch(self, model, positive, negative, threshold, cfg, uncond_zero_scale=0.0):
        g = AdaptiveGuider(model)
        g.set_conds(positive, negative)
        g.set_threshold(threshold)
        g.set_uncond_zero_scale(uncond_zero_scale)
        g.set_cfg(cfg)

        return (g,)


class Guider_PerpNegAG(comfy_extras.nodes_perpneg.Guider_PerpNeg):
    threshold_timestep = 0
    uz_scale = 0.0
    
    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_uncond_zero_scale(self, scale):
        self.uz_scale = scale

    def zero_cond(self, args):
        cond = args["cond_denoised"]
        x = args["input"]
        x -= x.mean()
        cond -= cond.mean()
        return x - (cond / cond.std() ** 0.5) * self.uz_scale
        
    def predict_noise(self, x, timestep, model_options={}, seed=None):
        cond = self.conds.get("positive")
        uncond = self.conds.get("negative")
        ts = timestep[0].item()
        if self.threshold_timestep > ts:
            if self.uz_scale > 0.0:
                model_options = model_options.copy()
                model_options["sampler_cfg_function"] = self.zero_cond
            return comfy.samplers.sampling_function(
                self.inner_model, x, timestep, uncond, cond, 1.0, model_options=model_options, seed=seed
            )
        self.threshold_timestep = 0

        # From comfy_extras.nodes_perpneg - Guider_PerpNeg
        # No need for calculating perp-neg when skipping negative
        empty_cond = self.conds.get("empty_negative_prompt")
        (cond_pred, uncond_pred, empty_cond_pred) = \
            comfy.samplers.calc_cond_batch(self.inner_model, [cond, uncond, empty_cond], x, timestep, model_options)
        cfg_result = comfy_extras.nodes_perpneg.perp_neg(x, cond_pred, uncond_pred, empty_cond_pred, self.neg_scale, self.cfg)

        if not self.threshold >= 1.0:
            sim = cos(cond_pred.reshape(1, -1), uncond_pred.reshape(1, -1)).item()
            if sim >= self.threshold:
                print(f"\nPerpNegAG: Cosine similarity {sim:.4f} exceeds threshold, setting CFG to 1.0")
                self.threshold_timestep = ts

        # From comfy_extras.nodes_perpneg - Guider_PerpNeg
        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": cfg_result,
                "cond": cond,
                "uncond": uncond,
                "model": self.inner_model,
                "uncond_denoised": uncond_pred,
                "cond_denoised": cond_pred,
                "sigma": timestep,
                "model_options": model_options,
                "input": x,
                # not in the original call in samplers.py:cfg_function, but made available for future hooks
                "empty_cond": empty_cond,
                "empty_cond_denoised": empty_cond_pred,}
            cfg_result = fn(args)

        return cfg_result

class PerpNegAGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "empty_conditioning": ("CONDITIONING", ),
                "threshold": ("FLOAT", {"default": 0.990, "min": 0.90, "max": 1.0, "step": 0.001, "round": 0.001}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "neg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {"uncond_zero_scale": ("FLOAT", {"default": 0.0, "max": 2.0, "step": 0.01})},
        }
    
    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"

    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, empty_conditioning, threshold, cfg, neg_scale, uncond_zero_scale=0.0):
        g = Guider_PerpNegAG(model)
        g.set_conds(positive, negative, empty_conditioning)
        g.set_threshold(threshold)
        g.set_uncond_zero_scale(uncond_zero_scale)
        g.set_cfg(cfg, neg_scale)

        return (g,)

NODE_CLASS_MAPPINGS = {
    "AdaptiveGuidance": AdaptiveGuidance,
    "PerpNegAdaptiveGuidanceGuider": PerpNegAGGuider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerpNegAdaptiveGuidanceGuider": "PerpNegAGGuider",
}
