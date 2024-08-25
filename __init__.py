import comfy.samplers
import comfy_extras.nodes_perpneg
import torch

cos = torch.nn.CosineSimilarity(dim=1)

# shared structure for adaptive guiders
class AdaptiveGuider(object):
    def __init__(self, model):
        super().__init__(model)  # This will initialize CFGGuider
        self.cfg_start_timestep = 1000.0
        self.threshold_timestep = 0
        self.uz_scale = 0.0
        self.threshold = 1.0
        self.original_cfg = 1.0

    def set_threshold(self, threshold, start_at):
        self.cfg_start_timestep = start_at
        self.threshold = threshold

    def set_uncond_zero_scale(self, scale):
        self.uz_scale = scale
        
    def zero_cond(self, args):
        cond = args["cond_denoised"]
        x = args["input"]
        x -= x.mean()
        cond -= cond.mean()
        return x - (cond / cond.std() ** 0.5) * self.uz_scale

    def set_cfg(self, cfg):
        self.cfg = cfg
        self.original_cfg = cfg

    def check_similarity(self, ts, cond_pred, uncond_pred):
        if self.threshold < 1.0:
            sim = cos(cond_pred.reshape(1, -1), uncond_pred.reshape(1, -1)).item()
            if sim >= self.threshold:
                print(f"AdaptiveGuider: Cosine similarity {sim:.4f} exceeds threshold, setting CFG to 1.0")
                self.threshold_timestep = ts
                self.cfg = 1.0
            else:
                self.cfg = self.original_cfg

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        ts = timestep[0].item()
        # Check if cfg is 1, if so, skip unnecessary calculations
        if self.cfg == 1.0:
            cond = self.conds.get("positive")
            uncond = self.conds.get("negative")
            # Directly return unconditioned noise prediction without guidance if cfg == 1.0
            return comfy.samplers.sampling_function(
                self.inner_model, x, timestep, uncond, cond, self.cfg, model_options=model_options, seed=seed
            )
        
        if ts >= self.cfg_start_timestep:
            current_cfg = self.original_cfg
        elif self.threshold_timestep > ts:
            current_cfg = 1.0
        else:
            conds = self.calc_conds(x, timestep, model_options)
            self.check_similarity(ts, conds[0], conds[1])
            current_cfg = self.cfg
            if self.threshold_timestep == 0:
                return self.calc_cfg(conds, x, timestep, model_options)

        if self.uz_scale > 0.0:
            model_options = model_options.copy()
            model_options["sampler_cfg_function"] = self.zero_cond
        
        cond = self.conds.get("positive")
        uncond = self.conds.get("negative")
        return comfy.samplers.sampling_function(
            self.inner_model, x, timestep, uncond, cond, current_cfg, model_options=model_options, seed=seed
        )

class Guider_AdaptiveGuidance(AdaptiveGuider, comfy.samplers.CFGGuider):
    def __init__(self, model):
        super().__init__(model)
    def calc_conds(self, x, timestep, model_options):
        cond = self.conds.get("positive")
        uncond = self.conds.get("negative")
        return comfy.samplers.calc_cond_batch(self.inner_model, [cond, uncond], x, timestep, model_options)

    def calc_cfg(self, conds, x, timestep, model_options):
        cond = self.conds.get("positive")
        uncond = self.conds.get("negative")
        cond_pred, uncond_pred = conds

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


class Guider_PerpNegAG(AdaptiveGuider, comfy_extras.nodes_perpneg.Guider_PerpNeg):
    def calc_conds(self, x, timestep, model_options):
        cond = self.conds.get("positive")
        uncond = self.conds.get("negative")
        empty_cond = self.conds.get("empty_negative_prompt")
        return comfy.samplers.calc_cond_batch(self.inner_model, [cond, uncond, empty_cond], x, timestep, model_options)

    def calc_cfg(self, conds, x, timestep, model_options):
        cond = self.conds.get("positive")
        uncond = self.conds.get("negative")
        empty_cond = self.conds.get("empty_negative_prompt")
        cond_pred, uncond_pred, empty_cond_pred = conds
        cfg_result = comfy_extras.nodes_perpneg.perp_neg(
            x, cond_pred, uncond_pred, empty_cond_pred, self.neg_scale, self.cfg
        )
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
                "empty_cond_denoised": empty_cond_pred,
            }
            cfg_result = fn(args)

        return cfg_result


class AdaptiveGuidanceGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "threshold": ("FLOAT", {"default": 0.990, "min": 0.90, "max": 1.0, "step": 0.0001, "round": 0.0001}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            },
            "optional": {
                "uncond_zero_scale": ("FLOAT", {"default": 0.0, "max": 2.0, "step": 0.01}),
                "cfg_start_pct": ("FLOAT", {"default": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"

    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, threshold, cfg, uncond_zero_scale=0.0, cfg_start_pct=0.0):
        cfg_start_timestep = model.get_model_object("model_sampling").percent_to_sigma(cfg_start_pct)
        g = Guider_AdaptiveGuidance(model)
        g.set_conds(positive, negative)
        g.set_threshold(threshold, cfg_start_timestep)
        g.set_uncond_zero_scale(uncond_zero_scale)
        g.set_cfg(cfg)

        return (g,)


class PerpNegAGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "empty_conditioning": ("CONDITIONING",),
                "threshold": ("FLOAT", {"default": 0.990, "min": 0.90, "max": 1.0, "step": 0.0001, "round": 0.0001}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "neg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "uncond_zero_scale": ("FLOAT", {"default": 0.0, "max": 2.0, "step": 0.01}),
                "cfg_start_pct": ("FLOAT", {"default": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"

    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(
        self,
        model,
        positive,
        negative,
        empty_conditioning,
        threshold,
        cfg,
        neg_scale,
        uncond_zero_scale=0.0,
        cfg_start_pct=0.0,
    ):
        cfg_start_timestep = model.get_model_object("model_sampling").percent_to_sigma(cfg_start_pct)
        g = Guider_PerpNegAG(model)
        g.set_conds(positive, negative, empty_conditioning)
        g.set_threshold(threshold, cfg_start_timestep)
        g.set_uncond_zero_scale(uncond_zero_scale)
        g.set_cfg(cfg, neg_scale)

        return (g,)


NODE_CLASS_MAPPINGS = {
    "AdaptiveGuidance": AdaptiveGuidanceGuider,
    "PerpNegAdaptiveGuidanceGuider": PerpNegAGGuider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveGuidance": "AdaptiveGuider",
    "PerpNegAdaptiveGuidanceGuider": "PerpNegAdaptiveGuider",
}
