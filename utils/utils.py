import torch
from pathlib import Path


def num_to_string(num, tol=2):
    """ convert numbers to strings.
    If round number, then return as integer.
    If decimal points are needed, replaced with character 'p' and round to `tol` 
    desimal places
    decimal points
    """
    is_int = (int(num) - num) == 0
    if is_int:
        return str(int(num))
    else:
        return f"{num:.{tol}f}".replace(".", "p")


def string_to_num(num):
    """
    Inverse of `num_to_str`. Read in string as a number, replacing `p` with a 
    decimal point. 
    """
    return float(num.replace("p", "."))


def parameters_checksum(model):
    if model is None:
        return 0
    sum = 0
    for p in model.parameters():
        sum += p.abs().sum().item()
    return sum


def filter_paths_png(paths):
    return [p for p in paths if Path(p).suffix == '.png']