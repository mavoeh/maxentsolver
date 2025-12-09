import torch
import torch.nn as nn

from .generative import GenMaxEnt


class MaxEntNodewisePL(nn.Module):
    """
    MaxEnt / Ising inference using nodewise pseudolikelihood maximization.

    Structure preserved from original version,
    but the fit() method uses *nodewise logistic regression*.
    """

    def __init__(self, n, device="cpu"):
        super().__init__()
        self.n = n
        self.device = device

        # Keep parameters in same format
        self.h = nn.Parameter(torch.zeros(n, device=device))
        self.J = nn.Parameter(torch.zeros(n, n, device=device))

        with torch.no_grad():
            self.J.data.fill_diagonal_(0.0)

    # ----------------------------------------------------
    # helper: symmetric J
    # ----------------------------------------------------
    def _symmetrize_J(self):
        with torch.no_grad():
            J = 0.5 * (self.J.data + self.J.data.t())
            J.fill_diagonal_(0.0)
            self.J.data = J

    # ----------------------------------------------------
    # helper: extract upper triangle
    # ----------------------------------------------------
    def _flatten_triu(self, M):
        idx = torch.triu_indices(self.n, self.n, offset=1, device=M.device)
        return M[idx[0], idx[1]]

    # ----------------------------------------------------
    # empirical marginals (unchanged)
    # ----------------------------------------------------
    def get_empirical_marginals(self, data):
        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        mean = data.mean(0)
        pair = (data.t() @ data) / data.shape[0]
        return mean, self._flatten_triu(pair)

    # ----------------------------------------------------
    # sampling through GenMaxEnt (unchanged)
    # ----------------------------------------------------
    def model_marginals(self, batch_size=100_000, num_sweeps=1_000, sequential=True, verbose=False):
        gen = GenMaxEnt(self.h.detach(), self.J.detach(), device=self.device)
        samples = gen.generate(num_samples=batch_size, num_sweeps=num_sweeps,
                               sequential=sequential, verbose=verbose)
        return self.get_empirical_marginals(samples)

    def fit(
        self,
        data,
        lr=0.05,
        steps=2000,
        l2=1e-3,
        patience=None,
        early_stop_tol=None,
        total_reports=1,
        verbose=True
    ):
        """
        Fully vectorized Nodewise Pseudolikelihood.
        Instead of n separate logistic regressions, 
        this trains all n nodes jointly in a single batched optimization.
        """

        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        N, n = data.shape

        # Convert Ising spins {-1,1} → {0,1}
        targets = (data + 1) * 0.5      # (N, n)

        # W is the coupling matrix (n,n) with zero diagonal
        # b is the biases (n)
        # They are learned directly and then copied to h,J afterward.
        W = nn.Parameter(torch.zeros(n, n, device=self.device))
        b = nn.Parameter(torch.zeros(n, device=self.device))

        # ensure no self-interactions
        with torch.no_grad():
            W.data.fill_diagonal_(0.0)

        opt = torch.optim.Adam([W, b], lr=lr)

        for step in range(steps):
            opt.zero_grad()

            # logits: (N, n)
            # Standard logistic regression: logits = X @ W + b
            logits = data @ W + b

            # BCE over all nodes and all samples
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)

            # L2 regularization (excluding diagonal)
            loss = loss + l2 * (W * (1 - torch.eye(n, device=self.device))).pow(2).sum()

            loss.backward()
            opt.step()

            # keep diagonal = 0
            with torch.no_grad():
                W.data.fill_diagonal_(0.0)

            if verbose and step % (steps // total_reports + 1) == 0:
                print(f"step {step:5d} | loss={loss.item():.6f}")

        # ----------------------------------------------------
        # convert logistic params → Ising params
        # PL: log P(s_i=1|s) = 2(h_i + Σ J_ij s_j)
        # logistic: log p = X @ W + b
        # => h_i = b_i / 2
        # => J_ij = W_ji / 2  (careful: orientation!)
        # ----------------------------------------------------
        with torch.no_grad():
            self.h.data = b / 2
            self.J.data = (W.T / 2)
            self.J.data.fill_diagonal_(0.0)
            self.J.data = 0.5 * (self.J.data + self.J.data.t())

        if verbose:
            print("\nVectorized nodewise PL finished.")

    # ----------------------------------------------------
    # unchanged
    # ----------------------------------------------------
    def interaction_matrix(self, numpy=True):
        M = self.J + torch.diag(self.h)
        return M.detach().cpu().numpy() if numpy else M
