from collections import defaultdict
import torch
import torch.nn as nn


from .mcmc import MaxEntMCMC
from .meanfield import MaxEntMeanField
from .pseudolikelihood import MaxEntPseudoLikelihood
from .nodewisepl import MaxEntNodewisePL
from ..utils import check_adjust_binary, binarize_data, NotBinaryError


class MaxEnt(nn.Module):
    _BACKENDS = {
        "mcmc": MaxEntMCMC,
        "meanfield": MaxEntMeanField,
        "mf": MaxEntMeanField,
        "mean_field": MaxEntMeanField,
        "pseudolikelihood": MaxEntPseudoLikelihood,
        "pl": MaxEntPseudoLikelihood,
        "pseudo_likelihood": MaxEntPseudoLikelihood,
        "nodewisepl": MaxEntNodewisePL,
    }

    def __init__(self, n: int, method: str = "pseudolikelihood", device=None, **kwargs):
        super().__init__()
        
        method = method.lower()
        if method not in self._BACKENDS:
            raise ValueError(f"method must be one of {list(self._BACKENDS.keys())}")
        
        if method == "mcmc":
            print("Warning: MCMC method is not correctly implemented yet. Every contribution is welcome at 'https://github.com/mavoeh/maxentsolver'.")
        
        self.n = n
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # THIS IS THE KEY: store the actual model as a regular attribute with NO special name
        self._model = self._BACKENDS[method](n=n, device=self._device, **kwargs)

        if isinstance(self._model, MaxEntMCMC):
            self.method = "MCMC"
        elif isinstance(self._model, MaxEntMeanField):
            self.method = "Mean-Field"
        elif isinstance(self._model, MaxEntPseudoLikelihood):
            self.method = "Pseudolikelihood"
        elif isinstance(self._model, MaxEntNodewisePL):
            self.method = "Nodewise Pseudolikelihood"

        # Register the real model as a sub-module so PyTorch sees its parameters
        self.add_module("core", self._model)

    # ==================== EXPLICITLY FORWARD EVERYTHING ====================
    def _symmetrize_J(self, J_param=None):
        return self._model._symmetrize_J(J_param)
    
    def fit(self, *args, **kwargs):
        try:
            args = list(args)
            args[0] = check_adjust_binary(args[0])
        except NotBinaryError:
            print("Data is mapped to binary {-1, +1} automatically.") if kwargs.get("verbose", True) else None
            args[0] = binarize_data(args[0], **kwargs)
        return self._model.fit(*args, **kwargs)
    
    def differentiable_fit(self, *args, **kwargs):
        if self.method != "meanfield":
            raise NotImplementedError("differentiable_fit() only available with Mean-Field")
        return self._model.differentiable_fit(*args, **kwargs)

    def model_marginals(self, *args, **kwargs):
        return self._model.model_marginals(*args, **kwargs)

    def interaction_matrix(self, **kwargs):
        return self._model.interaction_matrix(**kwargs)

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
    
    def energy(self, s):
        """
        s: (B, n) tensor with values in {-1, +1}
        Returns: (B,) energies
        """
        s = s.to(self._device)                       # (B, n)
        Jsym = self._symmetrize_J()                 # (n, n): symmetric, zero diagonal

        h_term = -torch.sum(self.core.h * s, dim=1)                    # (B,)
        J_term = -0.5 * torch.sum(s @ Jsym * s, dim=1)            # (B,)  ← crucial 0.5!

        return h_term + J_term
    
    def descend_to_minimum(self, s_init, max_steps=100_000, tol=0.0):
        s = s_init.clone().to(self._device)              # (B, n)
        B = s.shape[0]
        J = self._symmetrize_J()
        converged_mask = torch.zeros(B, dtype=torch.bool, device=self._device)

        for _ in range(max_steps):
            # Local field
            field = self.core.h + s @ J.t()                   # (B, n)  note: J.t() because s @ J.t() = (J @ s.t()).t()

            # ΔE for flipping i: exactly 2 * s_i * field_i
            delta_E = 2.0 * s * field                    # (B, n)

            # Find best (most negative) ΔE per sample
            best_dE, best_i = torch.min(delta_E, dim=1)  # (B,), (B,)

            # Which samples can still improve?
            active = best_dE < -tol                     # (B,)

            if not active.any():
                break

            # Only flip in active samples
            row_idx = torch.arange(B, device=self._device)[active]
            col_idx = best_i[active]

            # Perform the flip
            s[row_idx, col_idx] = -s[row_idx, col_idx]
            converged_mask |= ~active

        return s, converged_mask

    def find_minima(self, num_trials=1e8, batch_size=1e7, max_steps=1e6):
        
        counts = defaultdict(int)
        reps = {}           # canonical representative for each basin
        processed_batches = 0

        while processed_batches < num_trials:
            batch = min(batch_size, num_trials - processed_batches)
            s0 = torch.randn(batch, self.n, device=self._device).sign()
            s0[s0.abs() < 1e-8] = 1.0

            s_final, converged_mask = self.descend_to_minimum(s0, max_steps=max_steps)

            for idx, conf in enumerate(s_final):

                if converged_mask[idx] == False:
                    continue  # skip non-converged samples

                # Start with the arrived configuration
                candidate = conf
                key = tuple(candidate.cpu().numpy().tolist())

                # Now: key is the canonical key we will use
                if key not in reps:
                    reps[key] = candidate.clone()

                counts[key] += 1

            processed_batches += batch

        # Build results
        total = sum(counts.values())
        results = []
        for key in sorted(counts, key=counts.get, reverse=True):
            s = reps[key]
            e = self.energy(s.unsqueeze(0)).item()
            size = counts[key] / total
            results.append({
                'minimum': s.cpu(),
                'energy': e,
                'hits': counts[key],
                'approx_basin_size': size
            })

        # check if sizes sum to 1
        size_sum = sum(r['approx_basin_size'] for r in results)
        if abs(size_sum - 1.0) > 1e-4:
            print("Warning: basin sizes do not sum to 1 (sum =", size_sum, ")")

        return results

    # NO __getattr__ AT ALL → NO POSSIBLE RECURSION
    # If you need direct access to h/J, do: model.core.h, model.core.J
    # This is the price of absolute safety — and it's worth it.