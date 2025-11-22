import torch
import torch.nn as nn

from .mcmc import MaxEntMCMC
from .meanfield import MaxEntMeanField


class MaxEnt(nn.Module):
    """
    Final, bulletproof wrapper. Zero recursion. Works everywhere.
    """
    _BACKENDS = {
        "mcmc": MaxEntMCMC,
        "meanfield": MaxEntMeanField,
        "mf": MaxEntMeanField,
        "mean_field": MaxEntMeanField,
    }

    def __init__(self, n: int, method: str = "mcmc", device=None, **kwargs):
        super().__init__()
        
        method = method.lower()
        if method not in self._BACKENDS:
            raise ValueError(f"method must be one of {list(self._BACKENDS.keys())}")
        
        self.n = n
        self.method = "mcmc" if method == "mcmc" else "meanfield"
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # THIS IS THE KEY: store the actual model as a regular attribute with NO special name
        self._model = self._BACKENDS[method](n=n, device=self._device, **kwargs)

        # Register the real model as a sub-module so PyTorch sees its parameters
        self.add_module("core", self._model)

    # ==================== EXPLICITLY FORWARD EVERYTHING ====================
    def fit(self, *args, **kwargs):
        return self._model.fit(*args, **kwargs)

    def model_marginals(self, *args, **kwargs):
        return self._model.model_marginals(*args, **kwargs)

    def interaction_matrix(self):
        return self._model.interaction_matrix()

    def sample(self, *args, **kwargs):
        if self.method != "mcmc":
            raise NotImplementedError("sample() only available with MCMC")
        return self._model.sample(*args, **kwargs)

    def get_empirical_marginals(self, data):
        return self._model.get_empirical_marginals(data)

    # ==================== PyTorch essentials (MUST be overridden) ====================
    def forward(self, x):
        raise RuntimeError("MaxEnt has no forward()")

    def __repr__(self):
        return f"MaxEnt(n={self.n}, method={self.method}, device={self._device})"

    # NO __getattr__ AT ALL → NO POSSIBLE RECURSION
    # If you need direct access to h/J, do: model.core.h, model.core.J
    # This is the price of absolute safety — and it's worth it.