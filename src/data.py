from __future__ import annotations
import numpy as np
import torch
import math
import scipy
from typing import List, Tuple, Union, Any, Optional
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn.datasets import make_swiss_roll


########################
### Data Generation
########################

def make_training_data(config: Any) -> torch.Tensor:
    if config.dataset == 'funnel':
        training_distribution = Funnel(config.d)
        training_sample_position = training_distribution.sample(config.n_samples).to(config.device)
    elif config.dataset == 'mg25':
        training_distribution = MG25(config.d)
        training_sample_position = training_distribution.sample(config.n_samples).to(config.device)
    elif config.dataset == 'diamond':
        training_distribution = Diamond()
        training_sample_position = training_distribution.sample(config.n_samples).to(config.device)
    elif config.dataset == 'multimodal_swissroll':
        training_distribution = MultimodalSwissRoll()
        training_sample_position = training_distribution.sample(config.n_samples).to(config.device)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    return training_sample_position


class Funnel:
    def __init__(self, d: int, a: float = 1.0, b: float = 0.5) -> None:
        self.d = d
        self.a = a
        self.b = b

    def sample(self, n_samples: int) -> torch.Tensor:
        x1 = np.random.normal(0, self.a, size=n_samples)
        samples = np.zeros((n_samples, self.d))
        samples[:, 0] = x1
        variances = np.exp(2 * self.b * x1)
        samples[:, 1:] = np.random.normal(0, np.sqrt(variances)[:, np.newaxis], size=(n_samples, self.d - 1))
        return torch.tensor(samples, dtype=torch.float32)

    def get_log_pdf(self, x: torch.Tensor) -> float:
        x = np.asarray(x)
        cov_x1 = self.a**2
        cov_rest = np.exp(2 * self.b * x[0])
        log_pdf_x1 = multivariate_normal.logpdf(x[0], mean=0, cov=cov_x1)
        log_pdf_rest = sum(multivariate_normal.logpdf(x[i], mean=0, cov=cov_rest) for i in range(1, len(x)))
        return float(log_pdf_x1 + log_pdf_rest)
 
    def compute_NLL(self, samples: torch.Tensor) -> float:
        log_pdf_values = np.array([self.get_log_pdf(sample) for sample in samples])
        nll = -np.mean(log_pdf_values)
        return float(nll)



class MG25:
    def __init__(self, d: int) -> None:
        self.d = d
        self.cov_matrix = self.make_cov_matrix()
        self.means = self.make_means()

    def make_cov_matrix(self) -> np.ndarray:
        diag_elements = [0.01, 0.01] + [0.1] * (self.d - 2)
        return np.diag(diag_elements)

    def make_means(self) -> np.ndarray:
        means = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                mean = [i, j] + [0] * (self.d - 2)
                means.append(mean)
        return np.array(means)

    def sample(self, n_samples: int) -> torch.Tensor:
        samples = np.empty((n_samples, self.d))  
        mean_indices = np.random.choice(len(self.means), size=n_samples)  
        for index, mean_index in enumerate(mean_indices):
            mu = self.means[mean_index]
            sample = np.random.multivariate_normal(mean=mu, cov=self.cov_matrix)
            samples[index] = sample  
        return torch.tensor(samples, dtype=torch.float32)       

    def get_log_pdf(self, x: torch.Tensor) -> float:
        x = np.asarray(x)
        log_pdf_values = []
        for mean in self.means:
            log_pdf_values.append(multivariate_normal.logpdf(x, mean=mean, cov=self.cov_matrix))
        normalized_log_pdf = logsumexp(log_pdf_values) - np.log(25)
        return normalized_log_pdf

    def compute_NLL(self, samples: torch.Tensor) -> float:
        log_pdf_values = np.array([self.get_log_pdf(sample) for sample in samples])
        nll = -np.mean(log_pdf_values)
        return nll


class Diamond:
    def __init__(
        self,
        width: int = 3,
        bound: float = 0.5,
        noise: float = 0.04,
    ) -> None:
        self.width = width
        self.bound = bound
        self.noise = noise
        self.rotation_matrix = np.array([[1.0, -1.0], [1.0, 1.0]], dtype=np.float32) / np.sqrt(2.0)

    def sample(self, n_samples: int) -> torch.Tensor:
        means = np.array(
            [(x, y) for x in np.linspace(-self.bound, self.bound, self.width)
                     for y in np.linspace(-self.bound, self.bound, self.width)],
            dtype=np.float32,
        )
        means = means @ self.rotation_matrix
        covariance_factor = self.noise * np.eye(2, dtype=np.float32)

        index = np.random.choice(range(self.width ** 2), size=n_samples, replace=True)
        noise = np.random.randn(n_samples, 2).astype(np.float32)
        data = means[index] + noise @ covariance_factor
        return torch.from_numpy(data.astype("float32"))
    

class MultimodalSwissRoll:
    def __init__(
        self,
        noise: float = 0.2,
        multiplier: float = 0.01,
        offsets: list[list[float]] | None = None,
        weights: list[float] | None = None,
    ) -> None:
        self.noise = noise
        self.multiplier = multiplier
        self.offsets = offsets if offsets is not None else [[0.8, 0.8], [0.8, -0.8], [-0.8, -0.8], [-0.8, 0.8]]
        self.weights = weights if weights is not None else [0.2] * 5

    def sample(self, n_samples: int) -> torch.Tensor:
        idx = np.random.multinomial(n_samples, self.weights, size=1)[0]
        sr: list[np.ndarray] = []
        for k in range(5):
            cur = make_swiss_roll(int(idx[k]), noise=self.noise)[0][:, [0, 2]].astype("float32") * self.multiplier
            if k > 0:
                cur += np.array(self.offsets[k - 1], dtype=np.float32).reshape(-1, 2)
            sr.append(cur)
        data = np.concatenate(sr, axis=0)[np.random.permutation(n_samples)]
        return torch.from_numpy(data.astype("float32"))



########################
### Empirical Processing
########################
                   
def normalize(training_sample: torch.Tensor, rescale: float = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means = torch.mean(training_sample, dim=0)  
    std_devs = torch.std(training_sample, dim=0, unbiased=True)  
    normalized_sample = (training_sample - means) / (rescale*std_devs)
    return normalized_sample, means, std_devs

def unnormalize(normalized_sample: torch.Tensor, means: torch.Tensor, std_devs: torch.Tensor, rescale: float = 1) -> torch.Tensor:
    original_sample = (normalized_sample * rescale * std_devs) + means
    return original_sample

def empirical_mean_covar(sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = sample.mean(axis = 0)
    sample_centered = sample - mean
    covar = sample_centered.T @ sample_centered / (sample_centered.shape[0] - 1)
    return mean, covar

class empirical:
    def __init__(self, sample: torch.Tensor) -> None: 
        self.sample = sample
    def mean_covar(self) -> Tuple[torch.Tensor, torch.Tensor]: 
        return empirical_mean_covar(self.sample)

########################
### Metrics & Distances
########################

def wasserstein_w2(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor) -> float:
    mu1_np = mu1.cpu().numpy()
    sigma1_np = sigma1.cpu().numpy()
    mu2_np = mu2.cpu().numpy()
    sigma2_np = sigma2.cpu().numpy()
    diff_term = np.sum((mu1_np - mu2_np)**2)
    sqrt_sigma1 = scipy.linalg.sqrtm(sigma1_np).real
    sqrt_last = scipy.linalg.sqrtm(sqrt_sigma1 @ sigma2_np @ sqrt_sigma1).real
    sqrt_last_torch = torch.tensor(sqrt_last, device=mu1.device)
    return math.sqrt((diff_term + np.trace(sigma1_np + sigma2_np - 2 * sqrt_last)).item())

def w2(a: "empirical", b: "empirical") -> float: 
    return wasserstein_w2(*a.mean_covar(), *b.mean_covar())



class gaussian:
    def __init__(self, dimension: int, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        self.d: int = dimension
        self.device: torch.device = mu.device
        self._mu: torch.Tensor = mu
        self._sigma: torch.Tensor = sigma
        self._sq_sigma: torch.Tensor = torch.linalg.cholesky(self._sigma)

    def mean_covar(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._mu, self._sigma

    def generate_sample(self, size: int) -> torch.Tensor:
        self.sample = self._mu + torch.randn(size, self.d, device=self.device) @ self._sq_sigma.T
        return self.sample

    def to(self, device: Union[torch.device, str]) -> None:
        self.device = torch.device(device)
        self._mu = self._mu.to(self.device)
        self._sigma = self._sigma.to(self.device)
        self._sq_sigma = self._sq_sigma.to(self.device)
        if "sample" in self.__dict__:
            self.sample = self.sample.to(self.device)

    def compute_C0(self) -> torch.Tensor:
        # Compute the smallest eigenvalue of the Hessian of the Gaussian density (i.e. the inverse of largest eigenvalue of covariance matrix)
        eigenvalues: torch.Tensor = torch.linalg.eigvals(self._sigma)
        largest_eigenvalue: torch.Tensor = torch.max(torch.abs(eigenvalues))
        C0: torch.Tensor = 1.0 / largest_eigenvalue
        return C0

    def compute_L0(self) -> torch.Tensor:
        # Compute the largest eigenvalue of the Hessian of the Gaussian density (i.e. the inverse of smallest eigenvalue of covariance matrix)
        eigenvalues: torch.Tensor = torch.linalg.eigvals(self._sigma)
        smallest_eigenvalue: torch.Tensor = torch.min(torch.abs(eigenvalues))
        L0: torch.Tensor = 1.0 / smallest_eigenvalue
        return L0
