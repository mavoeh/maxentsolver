import torch
import torch.nn as nn


class MaxEntMeanField(nn.Module):
    def __init__(self, n, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.n = n
        self.device = device
        self.h = nn.Parameter(0.1 * torch.randn(n, device=device))
        self.J = nn.Parameter(0.1 * torch.randn(n, n, device=device))

    def _symmetrize_J(self):
        J = (self.J + self.J.t()) / 2
        J.fill_diagonal_(0)
        return J
    
    def get_empirical_marginals(self, data):
        data = torch.as_tensor(data, dtype=torch.float, device=self.device)
        mean = data.mean(0)
        cov = (data.t() @ data) / len(data)
        return mean, cov.triu(diagonal=1).flatten()

    def model_marginals(self, max_iter=100, tol=1e-7, **kwargs):
        m = torch.zeros_like(self.h)
        J = self._symmetrize_J()

        for it in range(max_iter):
            field = self.h + J @ m
            m_new = torch.tanh(field)
            if torch.norm(m_new - m) < tol:
                break
            m = 0.95 * m + 0.05 * m_new          # strong damping

        # Proper naive MF covariance
        cov = torch.outer(m, m) + torch.diag(m * (1 - m))
        return m, cov.triu(1).flatten()

    def fit(self, data, lr=1e-3, steps=4000, lambda_l1=1e-4):
        emp_mean, emp_cov = self.get_empirical_marginals(data)

        opt = torch.optim.Adam(self.parameters(), lr=lr)

        for step in range(steps):
            opt.zero_grad()
            model_mean, model_cov_flat = self.model_marginals()

            loss = 10.0 * ((emp_mean - model_mean)**2).sum() + \
                         ((emp_cov - model_cov_flat)**2).sum()   # down-weight cov

            loss += lambda_l1 * (self.h.abs().sum() + self.J.abs().sum())
            loss.backward()
            opt.step()

            if step % 500 == 0:
                print(f"Step {step:4d} | Loss {loss.item():.6f}")

        print("Final <m> range:", model_mean.abs().min().item(), "–", model_mean.abs().max().item())

    def interaction_matrix(self):
        return self._symmetrize_J().detach().cpu().numpy() + torch.diag(self.h).detach().cpu().numpy()

