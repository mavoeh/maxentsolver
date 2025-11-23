import torch
import torch.nn as nn


class MaxEntMeanField(nn.Module):
    def __init__(self, n, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.n = n
        self.device = device
        self.h = nn.Parameter(0.1 * torch.randn(n, device=device))
        self.J = nn.Parameter(0.1 * torch.randn(n, n, device=device))

    def _symmetrize_J(self, J_param=None):
        J = (self.J + self.J.t()) / 2 if J_param is None else (J_param + J_param.t()) / 2
        J = J - torch.diag(torch.diag(J))  # zero diagonal
        return J
    
    def _custom_step(self, params, grads, lr):
        # Simple gradient descent step
        return [(p - lr * g) for p, g in zip(params, grads)]

    
    def get_empirical_marginals(self, data):
        data = torch.as_tensor(data, dtype=torch.float, device=self.device)
        mean = data.mean(0)
        cov = (data.t() @ data) / len(data)
        return mean, cov.triu(diagonal=1).flatten()

    def model_marginals(self, max_iter=50_000, tol=1e-5, use_params=None, **kwargs):
        m = torch.tanh(self.h) if use_params is None else torch.tanh(use_params[0])
        J = self._symmetrize_J(use_params[1] if use_params else None)
        h = self.h if use_params is None else use_params[0]

        for it in range(max_iter):
            # Standard field
            field = h + J @ m
            
            # Onsager reaction term (the magic)
            reaction = J ** 2 @ ( m * (1 - m))
            
            m_new = torch.tanh(field - m * reaction)  # TAP equation

            error = torch.norm(m_new - m)
            
            if error < tol:
                break
                
            m = (m + m_new) / 2  # damping
        
        if it == max_iter - 1:
            print(f"Warning: Mean-field did not converge within max_iter, error = {error.item()}")

        # Covariance from susceptibility (much more accurate than naive MF)
        # χ₀⁻¹ = 1 - J (1 - m²)
        diag = 1 - m**2
        chi_inv = torch.eye(self.n, device=self.device) - J * diag
        chi = torch.inverse(chi_inv + 1e-6 * torch.eye(self.n, device=self.device))
        
        # Connected correlations
        cov_connected = chi - torch.diag(diag)
        cov = torch.outer(m, m) + cov_connected
        
        return m, cov.triu(1).flatten()

    def fit(self, data, lr=1e-3, steps=4000, lambda_l1=1e-4):
        emp_mean, emp_cov = self.get_empirical_marginals(data)

        opt = torch.optim.Adam(self.parameters(), lr=lr)

        for step in range(steps):
            opt.zero_grad()
            model_mean, model_cov_flat = self.model_marginals()

            loss = ((emp_mean - model_mean)**2).sum() + ((emp_cov - model_cov_flat)**2).sum()

            loss += lambda_l1 * (self.h.abs().sum() + self.J.abs().sum())
            loss.backward()
            opt.step()

            self.J.data = self._symmetrize_J(self.J.data)  # ensure symmetry

            if step % 500 == 0:
                print(f"Step {step:4d} | Loss {loss.item():.6f}")

        print("Final <m> range:", model_mean.abs().min().item(), "–", model_mean.abs().max().item())
    
    def differentiable_fit(self, data, lr=1e-3, steps=1000, lambda_l1=1e-4):
        emp_mean, emp_cov = self.get_empirical_marginals(data)

        params = [self.h, self.J]  # functional parameters

        for step in range(steps):
            # Forward passes using functional params
            model_mean, model_cov_flat = self.model_marginals(use_params=params)

            # Compute loss
            loss = ((emp_mean - model_mean)**2).sum() + ((emp_cov - model_cov_flat)**2).sum()
            loss += lambda_l1 * (params[0].abs().sum() + params[1].abs().sum())

            # Compute gradients w.r.t. functional params
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

            # Differentiable parameter update
            params = self._custom_step(params, grads, lr)
            params[1] = self._symmetrize_J(params[1])  # ensure symmetry

            if step % 500 == 0:
                print(f"Differentiable Step {step:4d} | Loss {loss.item():.6f}")
    
        params[1] = self._symmetrize_J(params[1])  # ensure symmetry
        self.h.data = params[0].data
        self.J.data = params[1].data

        # return functional parameters so outer loop can compute gradients
        return loss, params


    def interaction_matrix(self, numpy=True):
        total = self._symmetrize_J() + torch.diag(self.h)
        return total.detach().cpu().numpy() if numpy else total
