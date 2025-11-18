import torch
import math
import numpy as np


class beta_parametric:
    def __init__(self, a: float, final_time: float, beta_min: float, beta_max: float) -> None:
        self.a = a
        self.final_time = final_time
        self.beta_min = beta_min
        self.beta_max = beta_max
        if a == 0:
            self.delta = (beta_max - beta_min) / final_time
        else: 
            self.delta = (beta_max - beta_min) / (math.exp(self.a * final_time) - 1.)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if np.abs(self.a) < 1e-3:
            return self.beta_min + self.delta * t
        else:
            return self.beta_min + self.delta * (torch.exp(self.a * t) - 1.)

    def integrate(self, t: torch.Tensor) -> torch.Tensor:
        if np.abs(self.a) < 1e-3:
            return self.beta_min * t + 0.5 * self.delta * t**2
        else:
            return self.beta_min * t + self.delta * ((torch.exp(self.a * t) - 1) / self.a - t)

    def square_integrate(self, t: torch.Tensor) -> torch.Tensor:
        if np.abs(self.a) < 1e-3:
            return self.beta_min**2 * t + self.beta_min * self.delta * t**2 + (1.0 / 3) * self.delta**2 * t**3
        else:
            res = self.beta_min**2 * t + 2 * self.beta_min * self.delta * (torch.exp(self.a * t) / self.a - t)
            res += (self.delta)**2 * ((torch.exp(2 * self.a * t)) / (2 * self.a) - 2 * (torch.exp(self.a * t)) / (self.a) + t)
            res -= (2 * self.beta_min * self.delta / self.a - self.delta**2 * (3 / 2) * (1 / self.a))
            return res

    def change_a(self, a: float) -> None:
        self.a = a 
        if np.abs(self.a) < 1e-3: 
            self.delta = (self.beta_max - self.beta_min) / self.final_time 
        else:
            self.delta = (self.beta_max - self.beta_min) / (math.exp(self.a * self.final_time) - 1.)
