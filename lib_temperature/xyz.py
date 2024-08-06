from modules import scripts


def grid_reference():
    for data in scripts.scripts_data:
        if data.script_class.__module__ in (
            "scripts.xyz_grid",
            "xyz_grid.py",
        ) and hasattr(data, "module"):
            return data.module

    raise SystemError("Could not find X/Y/Z Plot...")


def xyz_support(cache: dict):

    def apply_field(field):
        def _(p, x, xs):
            cache.update({field: x})

        return _

    def choices_bool():
        return ["False", "True"]

    def choices_attention():
        return ["both", "self", "cross"]

    xyz_grid = grid_reference()

    extra_axis_options = [
        xyz_grid.AxisOption(
            "[Temperature] Temperature", float, apply_field("temperature")
        ),
        xyz_grid.AxisOption(
            "[Temperature] Attention",
            str,
            apply_field("attention"),
            choices=choices_attention,
        ),
        xyz_grid.AxisOption(
            "[Temperature] Dynamic Scale Temperature",
            str,
            apply_field("dynamic_temperature"),
            choices=choices_bool,
        ),
        xyz_grid.AxisOption(
            "[Temperature] Dynamic Scale Output",
            str,
            apply_field("dynamic_output"),
            choices=choices_bool,
        ),
    ]

    xyz_grid.axis_options.extend(extra_axis_options)
