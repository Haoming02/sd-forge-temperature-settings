"""
Credit: Extraltodeus
https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings

Modified by. Haoming02 to work with Forge
"""

from modules_forge.unet_patcher import UnetPatcher as Unet
# from ldm_patched.modules.sd import CLIP as Clip

from .const import EPSILON, SD_LAYER_DIMS, MODELS_BY_SIZE
# from functools import partial
from math import log, sqrt

import torch


def cv_temperature(input_tensor: torch.Tensor, auto_mode="normal") -> torch.Tensor:
    if "creative" not in auto_mode:
        temperature = torch.std(input_tensor)
    else:
        tensor_std = input_tensor.std()
        temperature = torch.std(torch.abs(input_tensor - tensor_std)) / tensor_std
        temperature = 1 / temperature
        del tensor_std

    if "squared" in auto_mode:
        temperature = temperature**2
    elif "sqrt" in auto_mode:
        temperature = temperature**0.5

    if not "reversed" in auto_mode:
        temperature = 1 / temperature

    return temperature


def should_scale(mname, lname, q2) -> bool:
    if not mname or mname == "CLIP":
        return False

    if mname != "Disabled" and lname in SD_LAYER_DIMS[mname]:
        if lname not in SD_LAYER_DIMS[mname]:
            return False
        else:
            return q2 != SD_LAYER_DIMS[mname][lname]

    return False


class _temperaturePatcher:

    def __init__(
        self,
        temperature: float,
        layer_name: str = "",
        model_name: str = "",
        eval_string: str = None,
        auto_temp: str = "disabled",
        Original_scale: int = 512,
        Target_scale_X: int = 512,
        Target_scale_Y: int = 512,
        rescale_adjust: float = 1.0,
        scale_before: bool = False,
        scale_after: bool = False,
    ):

        self.temperature = max(temperature, EPSILON)
        self.layer_name = layer_name
        self.model_name = model_name
        self.eval_string = eval_string
        self.auto_temp = auto_temp
        self.Original_scale = Original_scale
        self.Target_scale_X = Target_scale_X
        self.Target_scale_Y = Target_scale_Y
        self.rescale_adjust = rescale_adjust
        self.scale_before = scale_before
        self.scale_after = scale_after

    def pytorch_attention_with_temperature(
        self, q, k, v, extra_options, mask=None, attn_precision=None
    ):
        heads = (
            extra_options
            if isinstance(extra_options, int)
            else extra_options["n_heads"]
        )

        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

        if self.auto_temp == "disabled":
            extra_temperature = 1
        else:
            extra_temperature = cv_temperature(
                {"q_": q, "k_": k, "v_": v}[self.auto_temp[:2]], self.auto_temp
            )

        temperature_pre_scale = 1
        if self.scale_before:
            if should_scale(self.model_name, self.layer_name, q.size(-2)):
                ldim = SD_LAYER_DIMS[self.model_name][self.layer_name]

                if self.eval_string:
                    raise NotImplementedError()
                    # temperature_pre_scale = eval(self.eval_string)
                else:
                    temperature_pre_scale = log(q.size(-2), ldim)

            elif (self.Target_scale_X * self.Target_scale_Y) != self.Original_scale**2:
                temperature_pre_scale = log(
                    (self.Target_scale_X * self.Target_scale_Y) ** 0.5,
                    self.Original_scale,
                )

        temperature_scale = self.temperature / temperature_pre_scale

        scale = 1 / (sqrt(q.size(-1)) * temperature_scale * extra_temperature)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False, scale=scale
        )

        if self.scale_after:
            if should_scale(self.model_name, self.layer_name, q.size(-2)):
                ldim = SD_LAYER_DIMS[self.model_name][self.layer_name]

                if self.eval_string:
                    raise NotImplementedError()
                    # out *= eval(self.eval_string)
                else:
                    out *= log(q.size(-2), ldim)

            elif (self.Target_scale_X * self.Target_scale_Y) != self.Original_scale**2:
                out *= log(
                    (self.Target_scale_X * self.Target_scale_Y) ** 0.5,
                    self.Original_scale,
                )

        if self.rescale_adjust != 1.0:
            out *= self.rescale_adjust

        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class UnetTemperaturePatch:

    @staticmethod
    def patch(
        model: Unet,
        Temperature: float,
        Attention: str,
        Dynamic_Scale_Temperature: bool,
        Dynamic_Scale_Output: bool,
        Dynamic_Scale_Adjust: float = 1.0,
        eval_string: str = None,
        original: int = 512,
        target_width: int = 512,
        target_height: int = 512,
    ):
        dynamic_attention: bool = Dynamic_Scale_Temperature or Dynamic_Scale_Output

        if not dynamic_attention and Temperature == 1:
            print("\n[Temperature]: No patch applied...\n")
            return model

        if dynamic_attention and str(model.size) in MODELS_BY_SIZE:
            model_name = MODELS_BY_SIZE[str(model.size)]
            # print(f"\n[Temperature]: Model detected for scaling: {model_name}\n")
        else:
            model_name = "Disabled"
            if dynamic_attention:
                print("\n[Temperature]: Incompatible model for Dynamic Scale...\n")

        m = model.clone()
        levels = ("input", "middle", "output")
        layer_names = {f"{l}_{n}": True for l in levels for n in range(12)}

        for key, toggle in layer_names.items():
            current_level = key.split("_")[0]
            b_number = int(key.split("_")[1])

            patcher = _temperaturePatcher(
                Temperature,
                layer_name=key,
                model_name=model_name,
                eval_string=eval_string,
                rescale_adjust=Dynamic_Scale_Adjust,
                scale_before=Dynamic_Scale_Temperature,
                scale_after=Dynamic_Scale_Output,
                Original_scale=original,
                Target_scale_X=target_width,
                Target_scale_Y=target_height,
            )

            if Attention in ("both", "self"):
                m.set_model_attn1_replace(
                    patcher.pytorch_attention_with_temperature, current_level, b_number
                )
            if Attention in ("both", "cross"):
                m.set_model_attn2_replace(
                    patcher.pytorch_attention_with_temperature, current_level, b_number
                )

        return m


# ============================ #
# CLIP doesn't seem to work... #
# ============================ #

# class CLIPTemperaturePatch:

#     @staticmethod
#     def patch(clip: Clip, Temperature: float, Auto_temp: bool = False):
#         c = clip.clone()

#         def custom_optimized_attention(device, mask=None, small_input=True):
#             return _temperaturePatcher(
#                 temperature=Temperature,
#                 auto_temp="k_creative" if Auto_temp else "disabled",
#             ).pytorch_attention_with_temperature

#         def new_forward(self, x, mask=None, intermediate_output=None):
#             optimized_attention = custom_optimized_attention(
#                 x.device, mask=mask is not None, small_input=True
#             )

#             if intermediate_output is not None:
#                 if intermediate_output < 0:
#                     intermediate_output = len(self.layers) + intermediate_output

#             intermediate = None
#             for i, l in enumerate(self.layers):
#                 x = l(x, mask, optimized_attention)
#                 if i == intermediate_output:
#                     intermediate = x.clone()
#             return x, intermediate

#         if getattr(c.patcher.model, "clip_g", None):
#             c.patcher.add_object_patch(
#                 "clip_g.transformer.text_model.encoder.forward",
#                 partial(
#                     new_forward, c.patcher.model.clip_g.transformer.text_model.encoder
#                 ),
#             )

#         if getattr(c.patcher.model, "clip_l", None):
#             c.patcher.add_object_patch(
#                 "clip_l.transformer.text_model.encoder.forward",
#                 partial(
#                     new_forward, c.patcher.model.clip_l.transformer.text_model.encoder
#                 ),
#             )

#         return c


# class CLIPTemperatureWithScalePatch:

#     @staticmethod
#     def patch(
#         clip: Clip,
#         Temperature: float,
#         Dynamic_Scale_Temperature: bool,
#         Dynamic_Scale_Output: bool,
#         Original_scale: int = 512,
#         Target_scale_X: int = 512,
#         Target_scale_Y: int = 512,
#         Scale_Adjust: float = 1.0,
#     ):
#         c = clip.clone()

#         def custom_optimized_attention(device, mask=None, small_input=True):
#             return _temperaturePatcher(
#                 temperature=Temperature,
#                 Target_scale_X=Target_scale_X,
#                 Target_scale_Y=Target_scale_Y,
#                 Original_scale=Original_scale,
#                 rescale_adjust=Scale_Adjust,
#                 scale_before=Dynamic_Scale_Temperature,
#                 scale_after=Dynamic_Scale_Output,
#             ).pytorch_attention_with_temperature

#         def new_forward(self, x, mask=None, intermediate_output=None):
#             optimized_attention = custom_optimized_attention(
#                 x.device, mask=mask is not None, small_input=True
#             )

#             if intermediate_output is not None:
#                 if intermediate_output < 0:
#                     intermediate_output = len(self.layers) + intermediate_output

#             intermediate = None
#             for i, l in enumerate(self.layers):
#                 x = l(x, mask, optimized_attention)
#                 if i == intermediate_output:
#                     intermediate = x.clone()
#             return x, intermediate

#         if getattr(c.patcher.model, "clip_g", None):
#             c.patcher.add_object_patch(
#                 "clip_g.transformer.text_model.encoder.forward",
#                 partial(
#                     new_forward, c.patcher.model.clip_g.transformer.text_model.encoder
#                 ),
#             )

#         if getattr(c.patcher.model, "clip_l", None):
#             c.patcher.add_object_patch(
#                 "clip_l.transformer.text_model.encoder.forward",
#                 partial(
#                     new_forward, c.patcher.model.clip_l.transformer.text_model.encoder
#                 ),
#             )

#         return c
