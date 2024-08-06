"""
Credit: Extraltodeus
https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings

Modified by. Haoming02 to work with Forge
"""

EPSILON: float = 1e-16

SD_LAYER_DIMS: dict[str, dict[str, int]] = {
    "SD1": {
        "input_1": 4096,
        "input_2": 4096,
        "input_4": 1024,
        "input_5": 1024,
        "input_7": 256,
        "input_8": 256,
        "middle_0": 64,
        "output_3": 256,
        "output_4": 256,
        "output_5": 256,
        "output_6": 1024,
        "output_7": 1024,
        "output_8": 1024,
        "output_9": 4096,
        "output_10": 4096,
        "output_11": 4096,
    },
    "SDXL": {
        "input_4": 4096,
        "input_5": 4096,
        "input_7": 1024,
        "input_8": 1024,
        "middle_0": 1024,
        "output_0": 1024,
        "output_1": 1024,
        "output_2": 1024,
        "output_3": 4096,
        "output_4": 4096,
        "output_5": 4096,
    },
    "Disabled": {},
}

MODELS_BY_SIZE: dict[str, str] = {"1719049928": "SD1", "5134967368": "SDXL"}
