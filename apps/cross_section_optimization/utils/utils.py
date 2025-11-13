import sys, os, platform

__all__ = ["print_system_info", "get_system_info"]


def get_system_info() -> dict:
    return {
        "Python version": sys.version,
        "Operating System": f"{platform.system()} {platform.release()}",
        "Platform": platform.platform(),
        "Processor": platform.processor(),
        "Machine": platform.machine(),
        "CPU count": os.cpu_count(),
    }


def print_system_info() -> None:
    for key, value in get_system_info().items():
        print(f"{key}: {value}")