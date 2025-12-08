import torch
import torch.nn as nn

from .generative import GenMaxEnt


class MaxEntPseudoLikelihood(nn.Module):
    """
    MaxEnt / Ising inference using nodewise pseudolikelihood,
    rewritten to use CENTERED VARIABLES internally to avoid the
    well-known zero-mean degeneracy.

    Internally the model learns parameters (h', J') in centered
    coordinates x = s - mean(s), and converts them back to
    original Ising parameters (h, J) after training.

    All original methods remain available and work with (h, J).
    """

    def __init__(self, n, device="cpu"):
        super().__init__()
        self.n = n
        self.device = device

        # --- TRAINED PARAMETERS (centered model) ---
        self.h_prime = nn.Parameter(0.01 * torch.randn(n, device=device))
        self.J_prime = nn.Parameter(0.01 * torch.randn(n, n, device=device))

        with torch.no_grad():
            self.J_prime.data = (self.J_prime.data + self.J_prime.data.t()) / 2
            self.J_prime.data.fill_diagonal_(0.0)

        # --- RECONSTRUCTED ORIGINAL PARAMETERS (after training) ---
        self.register_buffer("mean", torch.zeros(n, device=device))
        self.register_buffer("h", torch.zeros(n, device=device))
        self.register_buffer("J", torch.zeros(n, n, device=device))

    # ======================================================================
    #  Utility / helpers
    # ======================================================================

    def _symmetrize_Jprime(self):
        """Ensure J' stays symmetric with zero diagonal."""
        with torch.no_grad():
            J = (self.J_prime.data + self.J_prime.data.t()) / 2
            J.fill_diagonal_(0.0)
            self.J_prime.data = J

    def _flatten_triu(self, M):
        """Vectorize upper triangle (i<j)."""
        idx = torch.triu_indices(self.n, self.n, offset=1, device=M.device)
        return M[idx[0], idx[1]]

    # ======================================================================
    #  Centering
    # ======================================================================

    def center_data(self, data):
        """
        Compute empirical means and return centered spins x = s - mean.
        """
        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        self.mean = data.mean(dim=0)
        return data - self.mean

    # ======================================================================
    #  Marginals
    # ======================================================================

    def get_empirical_marginals(self, data):
        """Return empirical mean and pairwise correlations."""
        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        mean = data.mean(0)
        pair = (data.t() @ data) / data.shape[0]
        return mean, self._flatten_triu(pair)

    def model_marginals(self, batch_size=100_000, num_sweeps=1_000,
                        sequential=True, verbose=False):
        """
        Estimate model marginals via sampling using original parameters (h, J).
        """
        gen = GenMaxEnt(self.h.detach(), self.J.detach(), device=self.device)
        samples = gen.generate(num_samples=batch_size,
                               num_sweeps=num_sweeps,
                               sequential=sequential,
                               verbose=verbose)
        return self.get_empirical_marginals(samples)

    # ======================================================================
    #  Pseudolikelihood loss (centered)
    # ======================================================================

    def pseudolikelihood_loss(self, X, l2):
        """
        Pseudolikelihood for centered variables x = s - m.

        Uses sign(X) for ±1 labels for the logistic regression.
        """
        X_sign = torch.sign(X)
        X_sign[X_sign == 0] = 1

        fields = X @ self.J_prime.T + self.h_prime
        logits = 2 * X_sign * fields

        log_probs = torch.nn.functional.logsigmoid(logits)
        loss = -log_probs.mean()
        loss += l2 * (self.J_prime ** 2).sum() + l2 * (self.h_prime ** 2).sum()

        return loss

    # ======================================================================
    #  Training
    # ======================================================================

    def fit(self,
            data,
            lr=1e-2,
            steps=2000,
            l2=0.0,
            patience=200,
            early_stop_tol=1e-6,
            total_reports=11,
            verbose=True):

        # ---- CENTER THE DATA ----
        X = self.center_data(data)

        opt = torch.optim.Adam(self.parameters(), lr=lr)

        best_loss = float("inf")
        best_h_prime = self.h_prime.detach().clone()
        best_J_prime = self.J_prime.detach().clone()
        no_improve = 0

        # reporting schedule
        report_steps = {int(steps * i / (total_reports - 1))
                        for i in range(total_reports)}

        for step in range(steps):
            opt.zero_grad()

            loss = self.pseudolikelihood_loss(X, l2)
            loss_value = loss.item()

            loss.backward()
            opt.step()
            self._symmetrize_Jprime()

            # --- early stopping ---
            if loss_value + early_stop_tol < best_loss:
                best_loss = loss_value
                best_h_prime = self.h_prime.detach().clone()
                best_J_prime = self.J_prime.detach().clone()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at step {step}")
                break

            if verbose and step in report_steps:
                print(f"Step {step:4d} | Loss = {loss_value:.6f}")

        # restore best centered params
        with torch.no_grad():
            self.h_prime.data.copy_(best_h_prime)
            self.J_prime.data.copy_(best_J_prime)

        if verbose:
            print("Restored best centered model.")
            print(f"Final centered loss = {best_loss:.6f}")

        # AFTER training: compute original Ising parameters
        self.finalize_original_parameters()

        return best_loss

    # ======================================================================
    #  Convert centered (h', J') → original (h, J)
    # ======================================================================

    def finalize_original_parameters(self):
        """
        Convert centered parameters back to original Ising parameters:

        h_i = h'_i - Σ_j J'_ij * mean_j
        J_ij = J'_ij
        """
        with torch.no_grad():
            self.J = self.J_prime.detach().clone()
            self.h = self.h_prime.detach().clone() - self.J @ self.mean

    # ======================================================================
    #  Original method preserved: interaction matrix
    # ======================================================================

    def interaction_matrix(self, numpy=True):
        """Return J + diag(h) from original parameterization."""
        M = self.J + torch.diag(self.h)
        return M.detach().cpu().numpy() if numpy else M
