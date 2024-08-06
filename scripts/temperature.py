from modules import scripts
import gradio as gr

from lib_temperature.nodes import UnetTemperaturePatch
from lib_temperature.xyz import xyz_support


class Temperature(scripts.Script):

    def __init__(self):
        self.xyzCache: dict = {}
        xyz_support(self.xyzCache)

    def title(self):
        return "Temperature Settings"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(open=False, label=self.title()):

            with gr.Row():
                enable = gr.Checkbox(label="Enable")
                attention = gr.Radio(
                    value="both", choices=["both", "self", "cross"], label="Attention"
                )

            temperature = gr.Slider(
                label="Temperature",
                minimum=0.5,
                maximum=1.5,
                step=0.05,
                value=1.0,
            )

            with gr.Row():
                dynamic_temperature = gr.Checkbox(
                    False, label="Dynamic Scale Temperature"
                )
                dynamic_output = gr.Checkbox(True, label="Dynamic Scale Output")

            self.paste_field_names = []
            self.infotext_fields = [
                (enable, "Temperature Enable"),
                (temperature, "Temperature"),
                (attention, "Temperature Attention"),
                (dynamic_temperature, "Dynamic Scale Temperature"),
                (dynamic_output, "Dynamic Scale Output"),
            ]

        for comp, name in self.infotext_fields:
            comp.do_not_save_to_config = True
            self.paste_field_names.append(name)

        return [enable, temperature, attention, dynamic_temperature, dynamic_output]

    def process_before_every_sampling(
        self,
        p,
        enable: bool,
        temperature: float,
        attention: str,
        dynamic_temperature: bool,
        dynamic_output: bool,
        *args,
        **kwargs,
    ):

        if len(self.xyzCache) > 0:
            enable = self.xyzCache.get("enable", enable)
            temperature = self.xyzCache.get("temperature", temperature)
            attention = self.xyzCache.get("attention", attention)
            dynamic_output = self.xyzCache.get("dynamic_output", dynamic_output)
            dynamic_temperature = self.xyzCache.get(
                "dynamic_temperature", dynamic_temperature
            )

            self.xyzCache.clear()

        if not enable:
            return p

        unet = p.sd_model.forge_objects.unet
        patched_unet = UnetTemperaturePatch.patch(
            unet,
            temperature,
            attention,
            dynamic_temperature,
            dynamic_output,
            1.0,
            None,
            1024 if getattr(p.sd_model, "is_sdxl", False) else 512,
            p.width,
            p.height,
        )
        p.sd_model.forge_objects.unet = patched_unet

        p.extra_generation_params.update(
            {
                "Temperature Enable": enable,
                "Temperature": temperature,
                "Temperature Attention": attention,
                "Dynamic Scale Temperature": dynamic_temperature,
                "Dynamic Scale Output": dynamic_output,
            }
        )
