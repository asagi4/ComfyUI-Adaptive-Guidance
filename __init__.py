import comfy.samplers
import comfy_extras.nodes_perpneg
import torch

cos = torch.nn.CosineSimilarity(dim=1)


# shared structure for adaptive guiders
class AdaptiveGuider(object):
    cfg_start_timestep = 1000.0
    threshold_timestep = 0
    uz_scale = 0.0

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

    def check_similarity(self, ts, cond_pred, uncond_pred):
        if not self.threshold >= 1.0:
            sim = cos(cond_pred.reshape(1, -1), uncond_pred.reshape(1, -1)).item()
            if sim >= self.threshold:
                print(f"AdaptiveGuider: Cosine similarity {sim:.4f} exceeds threshold, setting CFG to 1.0")
                self.threshold_timestep = ts

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        ts = timestep[0].item()
        if ts > self.cfg_start_timestep or self.threshold_timestep > ts or self.cfg == 1.0:
            if self.uz_scale > 0.0:
                model_options = model_options.copy()
                model_options["sampler_cfg_function"] = self.zero_cond
            cond = self.conds.get("positive")
            uncond = self.conds.get("negative")
            return comfy.samplers.sampling_function(
                self.inner_model, x, timestep, uncond, cond, 1.0, model_options=model_options, seed=seed
            )
        self.threshold_timestep = 0
        conds = self.calc_conds(x, timestep, model_options)
        self.check_similarity(ts, conds[0], conds[1])
        return self.calc_cfg(conds, x, timestep, model_options)


class Guider_AdaptiveGuidance(AdaptiveGuider, comfy.samplers.CFGGuider):
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


def project(a, b):
    dtype = a.dtype
    a, b = a.double(), b.double()
    b = torch.nn.functional.normalize(b, dim=[-1, -2, -3])
    a_par = (a * b).sum(dim=[-1, -2, -3], keepdim=True) * b
    a_orth = a - a_par
    return a_par.to(dtype), a_orth.to(dtype)


class AdaptiveProjectedGuidanceFunction:
    def __init__(self, momentum, eta, norm_threshold):
        self.eta = eta
        self.norm_threshold = norm_threshold
        self.current_step = 10000.0
        self.momentum = momentum
        self.running_average = 0.0

    def __call__(self, args):
        cond = args["cond_denoised"]
        uncond = args["uncond_denoised"]
        scale = args["cond_scale"]
        step = args["sigma"][0].item()
        x = args["input"]
        if self.current_step < step:
            self.current_step = 10000.0
            self.running_average = 0.0
        self.current_step = step

        diff = cond - uncond

        # I'm honestly not sure what this is supposed to do
        new_average = self.momentum * self.running_average
        self.running_average = diff + new_average
        diff = self.running_average

        if self.norm_threshold > 0.0:
            diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
            scale_factor = torch.minimum(torch.ones_like(diff), self.norm_threshold / diff_norm)
            diff = diff * scale_factor

        diff_parallel, diff_orthogonal = project(diff, cond)

        pred = cond + (scale - 1) * (diff_orthogonal + self.eta * diff_parallel)
        return x - pred


class AdaptiveProjectedGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {
                "momentum": ("FLOAT", {"default": -0.5, "min": -1.0, "max": 1.0, "step": 0.01}),
                "eta": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "norm_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "_for_testing"

    def apply(self, model, momentum=0.0, eta=1.0, norm_threshold=0.0):
        fn = AdaptiveProjectedGuidanceFunction(momentum, eta, norm_threshold)

        m = model.clone()
        m.set_model_sampler_cfg_function(fn)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "AdaptiveGuidance": AdaptiveGuidanceGuider,
    "PerpNegAdaptiveGuidanceGuider": PerpNegAGGuider,
    "AdaptiveProjectedGuidance": AdaptiveProjectedGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveGuidance": "AdaptiveGuider",
    "PerpNegAdaptiveGuidanceGuider": "PerpNegAdaptiveGuider",
}
