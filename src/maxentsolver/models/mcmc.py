import torch
import torch.nn as nn

class MaxEntMCMC(nn.Module):
    """Maximum Entropy (Ising) model using Gibbs sampling (scalable to n > 100)"""
    def __init__(self, n, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.n = n
        self.device = device
        self.h = nn.Parameter(0.1 * torch.randn(n, device=device))
        self.J = nn.Parameter(0.1 * torch.randn(n, n, device=device))  # Full matrix

    def _symmetrize_J(self):
        J = (self.J + self.J.t()) / 2
        J = J - torch.diag(torch.diag(J))  # Zero diagonal
        return J

    def _gibbs_step(self, samples):
        """One full Gibbs sweep. samples: (batch_size, n)"""
        J = self._symmetrize_J()                          # (n, n)
        # Compute conditional field for all spins and all samples at once
        conditional_field = self.h + samples @ J.t()       # (batch, n)
        probs = torch.sigmoid(conditional_field)          # P(s_i=1 | others)
        new_values = (torch.rand_like(probs) < probs).float()
        return new_values

    def sample(self, batch_size=1000, burnin=500, steps_per_sample=5):
        """Generate approximately independent samples from p(s)"""
        samples = (torch.rand(batch_size, self.n, device=self.device) > 0.5).float()
        for _ in range(burnin):
            samples = self._gibbs_step(samples)
        collected = []
        for _ in range(steps_per_sample):
            samples = self._gibbs_step(samples)
            collected.append(samples)
        return torch.cat(collected, dim=0)

    def model_marginals(self, batch_size=100000):
        samples = self.sample(batch_size=batch_size)
        mean = samples.mean(0)
        cov = (samples.t() @ samples) / len(samples)
        return mean, cov.triu(diagonal=1).flatten()

    def get_empirical_marginals(self, data):
        data = torch.as_tensor(data, dtype=torch.float, device=self.device)
        mean = data.mean(0)
        cov = (data.t() @ data) / len(data)
        return mean, cov.triu(diagonal=1).flatten()

    def fit(self, data, lr=1e-3, steps=4000, mcmc_steps=10, lambda_l1=1e-6):
        data = torch.as_tensor(data, dtype=torch.float, device=self.device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        emp_mean = data.mean(0)
        emp_cov = (data.t() @ data) / len(data)

        # Persistent chain
        samples = (torch.rand(256, self.n, device=self.device) > 0.5).float()

        for step in range(steps):
            opt.zero_grad()
            J = self._symmetrize_J()

            # Positive phase (data) - exact
            pos_lin = emp_mean
            pos_quad = emp_cov

            # Negative phase (model) - short MCMC from persistent chain
            for _ in range(mcmc_steps):
                samples = self._gibbs_step(samples)

            neg_lin = samples.mean(0)
            neg_quad = (samples.t() @ samples) / len(samples)

            # Moment matching loss (equivalent to -log-likelihood gradient)
            loss = ((pos_lin - neg_lin)**2).sum() + ((pos_quad - neg_quad)**2).sum()

            # L1 regularization
            loss += lambda_l1 * (self.h.abs().sum() + self.J.abs().sum())

            loss.backward()
            opt.step()

            if step % 500 == 0:
                print(f"Step {step:4d} | Loss: {loss.item():.4f}")

    def interaction_matrix(self):
        return self._symmetrize_J().detach().cpu().numpy() + torch.diag(self.h).detach().cpu().numpy()