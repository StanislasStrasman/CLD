from typing import Any, Callable, Tuple, Union
import math
import torch
from schedule import beta_parametric


class CriticallyDampedLangevin:
    def __init__(self, config: Any) -> None:
        # Initialize model hyperparameters and schedule from a config object.
        self.a = config.a
        self.epsilon = config.epsilon
        print(f"Using epsilon = {self.epsilon}")
        print(f"Using a = {self.a}")
        self.sigma = config.sigma
        self.device = config.device
        self.final_time = config.T
        self.beta_min = config.beta_min
        self.beta_max = config.beta_max
        self.beta = beta_parametric(0, self.final_time, self.beta_min, self.beta_max)
        self.v0_var = config.v0_var
        self.jitter = config.numerical_eps

    def mean(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  
        # compute mean factor for Denoising Score Matching
        batch_size = t.size(0)
        if u.dim() == 2:
            t = t.view(batch_size, 1)
        if u.dim() == 4:
            t = t.view(batch_size, 1, 1, 1)
        x, v = torch.chunk(u, 2, dim=1)
        
        tau = self.beta.integrate(t)
        exp_factor   = torch.exp(-self.a * tau)
        top_left     = 1 + self.a *  tau
        top_right    = (self.a ** 2) *  tau
        bottom_left  = - tau
        bottom_right = 1 - self.a  * tau

        mean_x  = top_left * x + top_right * v
        mean_v  = bottom_left * x + bottom_right * v
        mean_x *= exp_factor
        mean_v *= exp_factor
    
        mean_u = torch.cat((mean_x, mean_v), dim=1)
        return mean_u

    def mean_hsm(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  
        # compute mean factor for Hybrid Score Matching
        batch_size = t.size(0)
        if x.dim() == 2:
            t = t.view(batch_size, 1)
        if x.dim() == 4:
            t = t.view(batch_size, 1, 1, 1)
        tau = self.beta.integrate(t)
        exp_factor  = torch.exp(-self.a * tau)
        mean_x_h = exp_factor * (1 + self.a * tau) * x      
        mean_v_h = exp_factor * (-tau) * x           

        return torch.cat([mean_x_h, mean_v_h], dim=1)

    def compute_cholesky(self, var_x: torch.Tensor, cov_xv: torch.Tensor, var_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute Cholesky-like factors for a 2x2 covariance built from (var_x, cov_xv, var_v).
        l11 = torch.sqrt(var_x)
        l21 = cov_xv / l11
        l22 = torch.sqrt(var_v - l21**2)
        return l11, l21, l22        

    def cov(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
        # compute cov factor for Denoising Score Matching
        a = self.a
        epsilon = self.epsilon
        sigma = self.sigma
        tau = self.beta.integrate(t)
        
        if tau.dim() == 2:
            tau = tau.squeeze(1)
    
        exp_term = torch.exp(-2 * a * tau) / 4
    
        # Constant parts 
        var1_term1 = (1 / 4) * (5 * epsilon**2 / a + a * sigma**2)
        cov_term1  = (1 / 4) * (-2 * epsilon**2 / a**2)
        var2_term1 = (1 / 4) * (epsilon**2 + a**2 * sigma**2) / a**3
    
        # Time-dependent parts 
        var1_term2 = (-(5 + 2 * a * tau * (3 + a * tau)) * epsilon**2 - a**2 * (1 + 2 * a * tau * (1 + a * tau)) * sigma**2)/a 
        cov_term2  = 2 * ((epsilon + a * tau * epsilon)**2 + a**4 * (tau**2) * sigma**2) / a**2
        var2_term2 = (-(1 + 2 * a * tau * (1 + a * tau)) * epsilon**2 - a**2 * (1 + 2 * a * tau * (-1 + a * tau)) * sigma**2 )/ a**3
    
        var_x = var1_term1 + exp_term * var1_term2
        cov_xv  = cov_term1  + exp_term * cov_term2
        var_v = var2_term1 + exp_term * var2_term2
    
        return var_x + self.jitter, cov_xv, var_v + self.jitter

    def cov_hsm(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
        # compute cov factor for Hybrid Score Matching
        var_x, cov_xv, var_v = self.cov(t)
        tau = self.beta.integrate(t)
        if tau.dim() == 2:
            tau = tau.squeeze(1)
        exp2 = torch.exp(-2 * self.a * tau)  

        v      = self.v0_var
        add_x  = v * exp2 * (self.a**2 * tau)**2
        add_xv = v * exp2 * (self.a**2 * tau * (1 - self.a*tau))
        add_v  = v * exp2 * (1 - self.a*tau)**2

        return var_x  + add_x , cov_xv + add_xv, var_v  + add_v 

    def generate_forward(self, u: torch.Tensor, t: torch.Tensor, hsm: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
        # generate U_t|U_0
        if hsm:
            x0 = u
            mean = self.mean_hsm(x0, t).to(x0.device)  
            var_x, cov_xv, var_v = self.cov_hsm(t)
        else:
            u0       = u
            mean     = self.mean(u0, t).to(u0.device)
            var_x, cov_xv, var_v = self.cov(t)

        l11, l21, l22 = self.compute_cholesky(var_x, cov_xv, var_v)

        extra_dims = mean.dim() - 1  
        shape = (l11.size(0),) + (1,) * extra_dims  
        l11 = l11.view(shape)
        l21 = l21.view(shape)
        l22 = l22.view(shape)

        eps = torch.randn_like(mean, device=mean.device)    
        rx, rv = eps.chunk(2, dim=1)                    

        noise_x = l11 * rx
        noise_v = l21 * rx + l22 * rv
        noise   = torch.cat([noise_x, noise_v], dim=1)
        return mean + noise, noise, eps
     
    def sample_final(self, num_samples: int, dim: Union[int, Tuple[int, int, int]]) -> torch.Tensor:  
        # sample from the stationary distribution
        a = self.a
        epsilon = self.epsilon
        sigma = self.sigma
        device = self.device

        if isinstance(dim, int):
            shape = (num_samples, 2 * dim)
        else:
            C, H, W = dim
            shape = (num_samples, 2 * C, H, W)
    
        var_x = torch.tensor((5 * epsilon**2 / a + a * sigma**2) / (4.0 ), device=device, dtype=torch.float32)
        cov_xv = torch.tensor((-2 * epsilon**2 / a**2) / (4.0), device=device, dtype=torch.float32)
        var_v = torch.tensor((epsilon**2 + a**2 * sigma**2) / (4.0 * a**3), device=device, dtype=torch.float32)
    
        l11, l21, l22 = self.compute_cholesky(var_x, cov_xv, var_v)
    
        z = torch.randn(shape, device=device)
        z_x, z_v = torch.chunk(z, 2, dim=1)
    
        sample_x = l11 * z_x
        sample_v = l21 * z_x + l22 * z_v
    
        return torch.cat((sample_x, sample_v), dim=1)

    def Euler_Maruyama_discr_step(self, u: torch.Tensor, score_xv: torch.Tensor, t: torch.Tensor, step_size: float, is_cld: bool) -> torch.Tensor:
        # One Euler–Maruyama update for (x, v) at time t
        beta_t = self.beta(t)
        top_left = 0.0
        top_right = self.a ** 2 * beta_t
        bottom_left = -1 * beta_t
        bottom_right = -2 * self.a * beta_t
        sigma_x = self.epsilon
        sigma_v = self.sigma

        x, v = torch.chunk(u, 2, dim=1)
        if is_cld:
            score_v = score_xv
            score_x = torch.zeros_like(x)
        else:
            score_x, score_v = torch.chunk(score_xv, 2, dim=1)

        drift_x = - top_left * x - top_right * v + sigma_x**2 * score_x * beta_t
        drift_v = - bottom_left * x - bottom_right * v + sigma_v**2 * score_v * beta_t

        noise_x =  math.sqrt(step_size * beta_t) * sigma_x * torch.randn_like(x)
        noise_v = math.sqrt(step_size * beta_t) * sigma_v * torch.randn_like(v) 

        x = x + drift_x * step_size + noise_x
        v = v + drift_v * step_size + noise_v

        return torch.cat([x, v], dim=1)

    def Euler_Maruyama_discr_sampler(self,
                                     init: torch.Tensor,
                                     score_theta: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                     num_steps: int,
                                     epsilon_s: float,
                                     is_cld: bool = False) -> torch.Tensor:
        # Run Euler–Maruyama sampling for a given number of steps
        u = init.clone()
        times = self.get_time_steps(num_steps, epsilon_s=epsilon_s)  
        dt = (times[0] - times[1]).item()
    
        with torch.no_grad():
            for t in times:
                tau = t.expand(u.shape[0])
                score_xv = score_theta(u, tau)
                u = self.Euler_Maruyama_discr_step(
                    u, score_xv, t, dt, is_cld
                )
    
        x_bar, _ = torch.chunk(u, 2, dim=1)
        return x_bar
        
    def leapfrog_step(self, u: torch.Tensor, score_xv: torch.Tensor, t: torch.Tensor, step_size: float, is_cld: bool, score_theta: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        # One leapfrog step
        epsilon = self.epsilon
        sigma = self.sigma
    
        beta_n = self.beta(t)
        beta_half = self.beta(t + 0.5 * step_size)
    
        x, v = torch.chunk(u, 2, dim=1)
    
        if is_cld:
            score_v = score_xv
            score_x = torch.zeros_like(x)
        else:
            score_x, score_v = torch.chunk(score_xv, 2, dim=1)
    
        v = v + 0.5 * step_size * (
            beta_n * (x + 2 * a * v + sigma**2 * score_v)
        )
    
        x = x + 0.5 * step_size * (
            beta_n * (-a**2 * v + epsilon**2 * score_x)
        )
    
        noise_x = math.sqrt(step_size * beta_n) * epsilon * torch.randn_like(x)
        noise_v = math.sqrt(step_size * beta_n) * sigma * torch.randn_like(v)
        x = x + noise_x
        v = v + noise_v
    
        u_new = torch.cat([x, v], dim=1)
        score_xv_new = score_theta(u_new, t + 0.5 * step_size)
    
        if is_cld:
            score_v_new = score_xv_new
            score_x_new = torch.zeros_like(x)
        else:
            score_x_new, score_v_new = torch.chunk(score_xv_new, 2, dim=1)
    
        x = x + 0.5 * step_size * (
            beta_half * (-a**2 * v + epsilon**2 * score_x_new)
        )
    
        v = v + 0.5 * step_size * (
            beta_half * (x + 2 * a * v + sigma**2 * score_v_new)
        )
    
        return torch.cat([x, v], dim=1)
    
    def leapfrog_sampler(self, init: torch.Tensor, score_theta: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], num_steps: int, epsilon_s: float, is_cld: bool = False) -> torch.Tensor:
        # Run leapfrog-based sampling for a given number of steps
        u = init.clone()
        times = self.get_time_steps(num_steps, epsilon_s=epsilon_s)
        dt = (times[0] - times[1]).item()
    
        with torch.no_grad():
            for t in times:
                score_xv = score_theta(u, t)
                u = self.leapfrog_step(u, score_xv, t, dt, is_cld, score_theta)
    
        x_bar, _ = torch.chunk(u, 2, dim=1)
        return x_bar

    def get_time_steps(self, num_steps: int, epsilon_s: float = 1e-3) -> torch.Tensor:
        # Construct a descending linear schedule of time points in [epsilon_s, 1]
        N = num_steps
        i = torch.arange(N, device=self.device)          
        t_i = 1 + (i/(N-1))*(epsilon_s - 1)
        return t_i
