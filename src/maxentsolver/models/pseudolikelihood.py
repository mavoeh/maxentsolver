import torch
import torch.nn as nn

from .generative import GenMaxEnt


class MaxEntPseudoLikelihood(nn.Module):
    """
    MaxEnt / Ising inference using nodewise pseudolikelihood maximization.
    
    Features:
        - Symmetric J with zero diagonal
        - Early stopping + best-model restore
        - Configurable reporting (total_reports)
        - L2 regularization passed to fit()
    """

    def __init__(self, n, device="cpu"):
        super().__init__()
        self.n = n
        self.device = device

        self.h = nn.Parameter(0.01 * torch.randn(n, device=device))
        self.J = nn.Parameter(0.01 * torch.randn(n, n, device=device))

        with torch.no_grad():
            self.J.data = (self.J.data + self.J.data.t()) / 2
            self.J.data.fill_diagonal_(0.0)

    def _symmetrize_J(self):
        """Force J to remain symmetric with 0 diagonal."""
        with torch.no_grad():
            J = (self.J.data + self.J.data.t()) / 2
            J.fill_diagonal_(0.0)
            self.J.data = J

    def _flatten_triu(self, M):
        """Vectorize upper triangle (i<j) of a matrix."""
        idx = torch.triu_indices(self.n, self.n, offset=1, device=M.device)
        return M[idx[0], idx[1]]

    def get_empirical_marginals(self, data):
        """Return empirical mean and pairwise correlations."""
        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        mean = data.mean(0)
        pair = (data.t() @ data) / data.shape[0]
        return mean, self._flatten_triu(pair)

    def model_marginals(self, batch_size=100_000, num_sweeps=1_000, sequential=True):
        """Estimate model marginals via sampling."""
        gen = GenMaxEnt(self.h.detach(), self.J.detach(), device=self.device)
        samples = gen.generate(num_samples=batch_size, num_sweeps=num_sweeps, sequential=sequential)
        return self.get_empirical_marginals(samples)

    def pseudolikelihood_loss(self, data, l2):
        """
        Negative log pseudolikelihood:
            log P(s_i | s_-i) = log σ( 2 s_i (h_i + Σ_j J_ij s_j) )
        """
        s = data.to(self.device)  # (N, n)

        fields = s @ self.J.T + self.h  # shape: (N, n)

        logits = 2 * s * fields
        log_probs = torch.nn.functional.logsigmoid(logits)

        loss = -log_probs.mean()
        loss += l2 * (self.J ** 2).sum()
        return loss

    def fit(
        self,
        data,
        lr=1e-2,
        steps=2000,
        l2=0.0,
        patience=200,
        early_stop_tol=1e-6,
        total_reports=10,
        verbose=True
    ):

        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        best_loss = float("inf")
        best_h = self.h.detach().clone()
        best_J = self.J.detach().clone()
        no_improve = 0

        # reporting schedule
        report_steps = set()
        for i in range(total_reports):
            report_steps.add(int(steps * i / (total_reports - 1)))

        for step in range(steps):
            opt.zero_grad()

            loss = self.pseudolikelihood_loss(data, l2)
            loss_value = loss.item()

            loss.backward()
            opt.step()

            # Enforce symmetric J
            self._symmetrize_J()

            # ------------- Early stopping bookkeeping ----------------
            if loss_value + early_stop_tol < best_loss:
                best_loss = loss_value
                best_h = self.h.detach().clone()
                best_J = self.J.detach().clone()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at step {step} (no improvement for {patience} steps)")
                break

            # ------------- Reporting ----------------
            if verbose and step in report_steps:
                print(f"Step {step:4d} | Loss = {loss_value:.6f}")

        # Restore best parameters
        with torch.no_grad():
            self.h.data.copy_(best_h)
            self.J.data.copy_(best_J)

        if verbose:
            print("Restored best model.")
            print(f"Final loss = {best_loss:.6f}")

        return best_loss

    def interaction_matrix(self, numpy=True):
        """Return J + diag(h)."""
        M = self.J + torch.diag(self.h)
        return M.detach().cpu().numpy() if numpy else M
