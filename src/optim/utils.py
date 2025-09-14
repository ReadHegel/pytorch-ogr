from torch import Tensor
from torch import cat, split

def flat_tensor_list(params: list[Tensor]):
    return cat([
        param.flatten() for param in params
    ]) 

def restore_tensor_list(flat_params, params_sizes, params_shapes):
    splited = split(flat_params, params_sizes)
    return [s.view(shape) for s, shape in zip(splited, params_shapes)]
