import importlib.util


def is_module_available(module_name):
    return importlib.util.find_spec(module_name) is not None


__hasmatplotlib__ = is_module_available("matplotlib")


config = dict(
    num_MC_samples=1000,
)

__all__ = ["__hasmatplotlib__", "config"]
