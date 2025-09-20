import torch 
from torch import Tensor
from numpy import linspace 

class Linesearch(): 
    def __init__(self, f, num_steps: int = 50):
        self.f = f
        self.num_steps = num_steps

    def perform_search(self, init_param: Tensor, direction: Tensor) -> Tensor: 
        """ returns the step found by line search """
        argmin = init_param
        min_value = self.f(init_param)

        for alpha in linspace(0, 2, num=self.num_steps):
            new_param = init_param + alpha * direction
            new_value = self.f(new_param) 
            
            if new_value < min_value: 
                argmin =  new_param
                min_value = new_value 

        return argmin - init_param 

