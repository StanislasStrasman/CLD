import torch
from typing import Any, Callable, Tuple

####### Loss Functions ########

#### Hybrid Score Matching Loss #####

class loss_hsm:
    def __init__(self, score_theta: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], sde: Any, eps: float) -> None:
        self.score_theta = score_theta
        self.sde         = sde
        self.eps         = eps

    def __call__(self, x0: torch.Tensor, is_cld: bool = False) -> torch.Tensor:
        batch = x0.shape[0]
        device   = x0.device

        time_tau = torch.rand(batch, device=device) * (self.sde.final_time - self.eps) + self.eps
        #time_tau = torch.rand(batch, 1, device=device) * (self.sde.final_time - self.eps) + self.eps   ## for UNet

        u_t, noise, eps = self.sde.generate_forward(x0, time_tau, hsm=True)
        _ , eps_v = torch.chunk(eps, 2, dim=1)       
        
        score = self.score_theta(u_t, time_tau)
        score_x , score_v = torch.chunk(score, 2, dim=1)
        var_x, cov, var_v = self.sde.cov_hsm(time_tau)
        noise_x, noise_v = torch.chunk(noise, 2, dim=1)
 
        det = var_x * var_v - cov ** 2

        extra_dims = noise.dim() - 1  
        shape = (noise.size(0),) + (1,) * extra_dims  
        var_x = var_x.view(shape)
        cov = cov.view(shape)
        var_v = var_v.view(shape)
        det = det.view(shape)
        beta_t = self.sde.beta(time_tau)

        if is_cld:
            pred =  -(1/torch.sqrt(var_x/det))*score -eps_v
            w = 1 
            loss = torch.mean(torch.sum(w * pred **2, dim=1))
        else:
            pred_x = (- var_x *score_x - cov*score_v) -noise_x
            pred_v = (-cov *score_x - var_v*score_v) -noise_v
            loss_x =  (var_v * pred_x - cov*pred_v)
            loss_v = (-cov*pred_x + var_x*pred_v)
            weighted = (1/(var_x*var_v-cov**2))*(loss_x**2 + loss_v**2)
            loss = torch.mean(torch.sum(weighted, dim=1))

        return loss


#### Denosing Score Matching Loss #####

class loss_conditional:
    def __init__(self, score_theta: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], sde: Any, eps: float) -> None:
        self.score_theta = score_theta
        self.sde = sde
        self.eps = eps 

    def __call__(self, u: torch.Tensor, weight: float = 1) -> torch.Tensor:
        time_tau = torch.rand(u.shape[0], device=u.device) * (self.sde.final_time - self.eps) + self.eps
        x_tau, noise = self.sde.generate_forward(u, time_tau)
        score =  self.score_theta(x_tau, time_tau) 
        cov_t = self.sde.cov(time_tau)
        target = compute_target(cov_t, noise)
        loss = torch.mean(torch.sum((score - target)**2, axis=1))
        return loss
    

def compute_target(cov_t: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], noise: torch.Tensor) -> torch.Tensor:
    var1, cov, var2 = cov_t
    noise_x, noise_v = torch.chunk(noise, 2, dim=1)
    det = var1 * var2 - cov ** 2

    extra_dims = noise.dim() - 1  
    shape = (noise.size(0),) + (1,) * extra_dims  
    var1 = var1.view(shape)
    cov = cov.view(shape)
    var2 = var2.view(shape)
    det = det.view(shape)

    inv_x = (var2 * noise_x - cov * noise_v) / det
    inv_v = (-cov * noise_x + var1 * noise_v) / det

    return -torch.cat([inv_x, inv_v], dim=1)
