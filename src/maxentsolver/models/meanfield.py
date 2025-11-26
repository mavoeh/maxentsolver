import torch
import torch.nn as nn


class MaxEntMeanField(nn.Module):
    def __init__(self, n, device='cpu', anderson_m=5, anderson_beta=1.0):
        super().__init__()
        self.n = n
        self.device = device
        self.h = nn.Parameter(0.1 * torch.randn(n, device=device))
        self.J = nn.Parameter(0.1 * torch.randn(n, n, device=device))
        with torch.no_grad():
            self.J.data = (self.J.data + self.J.data.t()) / 2
            self.J.data.fill_diagonal_(0.0)
        self.anderson_m = anderson_m
        self.anderson_beta = anderson_beta  # mixing for Anderson extrapolation

    def _symmetrize_J(self, J_param=None):
        J = self.J if J_param is None else J_param
        Js = (J + J.t()) / 2
        Js = Js - torch.diag(torch.diag(Js))
        return Js

    def _flatten_triu(self, M):
        idx = torch.triu_indices(self.n, self.n, offset=1, device=M.device)
        return M[idx[0], idx[1]]

    def get_empirical_marginals(self, data):
        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        mean = data.mean(0)
        emp_pair = (data.t() @ data) / data.shape[0]
        return mean, self._flatten_triu(emp_pair)

    def _anderson(self, f, x0, tol=1e-5, max_iter=2_000, damping=0.5):
        """
        Anderson acceleration for fixed-point iteration x = f(x).
        Fully detached; does NOT track autograd through iterations.
        """
        m = self.anderson_m
        beta = self.anderson_beta

        x = x0.clone()  # detached to avoid autograd issues
        dtype = x.dtype
        Fmat = torch.zeros(x.numel(), m, device=self.device, dtype=dtype)
        f_hist = 0
        last_err = None

        for k in range(max_iter):
            # Fixed-point update
            x_new = f(x)
            res = (x_new - x).detach()
            err = torch.norm(res) / self.n
            last_err = err.item()

            if err < tol:
                return x_new, k + 1, err

            # Update history buffers
            idx = f_hist % m
            Fmat[:, idx].copy_(res.flatten())
            f_hist += 1
            hist_len = min(f_hist, m)

            if hist_len >= 2:
                Fsub = Fmat[:, :hist_len]  # shape (n, r)
                FtF = Fsub.T @ Fsub
                rhs = Fsub.T @ res.flatten()
                reg = 1e-6 * torch.eye(hist_len, device=self.device, dtype=dtype)
                try:
                    # solve FtF * alpha = rhs via Cholesky (stable)
                    L = torch.linalg.cholesky(FtF + reg)
                    alpha = torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze(1)
                    extrap = (Fsub @ alpha).reshape_as(x)
                    x_and = x + beta * extrap
                    x = damping * x + (1 - damping) * x_and
                except RuntimeError:
                    # fallback to normal damped update if Cholesky fails
                    x = damping * x + (1 - damping) * x_new
            else:
                # standard damped update
                x = damping * x + (1 - damping) * x_new

        return x, max_iter, last_err

    def model_marginals(self, max_iter=2_000, tol=1e-5, use_params=None, damping=0.5):
        """
        Pure TAP with Anderson acceleration. No differentiable unroll.
        Returns:
        m: final magnetizations (detached)
        cov_flat: flattened upper-triangle of covariance
        """
        h = self.h if use_params is None else use_params[0]
        J = self._symmetrize_J(None if use_params is None else use_params[1])

        # initial guess
        m0 = torch.tanh(h)

        def tap_step(m):
            field = h + J @ m
            reaction = (J * J) @ (1 - m**2)
            return torch.tanh(field - m * reaction)

        # Anderson acceleration to fixed point
        m_final, iters, err = self._anderson(tap_step, m0, tol=tol, max_iter=max_iter, damping=damping)

        if iters == max_iter:
            print(f"Warning: Anderson did not converge within {max_iter} iterations. Final error: {err:.2e}")

        m = m_final

        # covariance via susceptibility
        diag_vec = 1 - m**2
        eps = 1e-8
        D_inv = torch.diag(1.0 / (diag_vec + eps))
        I = torch.eye(self.n, device=self.device)
        A = D_inv - J + 1e-6 * I

        # solve for chi
        chi = torch.linalg.solve(A, I)

        cov = torch.outer(m, m) + chi

        return m, self._flatten_triu(cov)

    def fit(
        self, 
        data, 
        lr=1e-3, 
        steps=4000, 
        lambda_l1=1e-4, 
        max_iter=2_000, 
        tol=1e-5, 
        damping=0.5,
        verbose=True,
        early_stop_tol=1e-6,
        patience=200,
        total_reports=11,
    ):
        """
        Improved fit() with:
        - adaptive printing (5 total prints)
        - verbose argument
        - early stopping
        - restoring the best model
        """

        # Empirical marginals
        emp_mean, emp_cov = self.get_empirical_marginals(data)

        opt = torch.optim.Adam(self.parameters(), lr=lr)

        # For early stopping + model restoration
        best_loss = float("inf")
        best_h = self.h.detach().clone()
        best_J = self.J.detach().clone()
        no_improve = 0

        report_steps = set()
        for i in range(total_reports):  
            report_steps.add(int(steps * i / (total_reports - 1)))

        curr_tol = 1.0  # start with a looser tolerance
        for step in range(steps):

            if step < patience:
                curr_tol = max(tol, curr_tol * 0.9)
            else:
                curr_tol = tol

            opt.zero_grad()
            model_mean, model_cov_flat = self.model_marginals(
                max_iter=max_iter, tol=curr_tol, damping=damping
            )

            # Loss = moment matching
            loss = ((emp_mean - model_mean)**2).sum() + ((emp_cov - model_cov_flat)**2).sum()
            loss += lambda_l1 * (self.h.abs().sum() + self.J.abs().sum())

            loss.backward()
            opt.step()

            # enforce symmetry
            self.J.data = self._symmetrize_J(self.J.data)

            # ---- Early stopping bookkeeping ----
            loss_value = loss.item()

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

            # ---- Adaptive printing (exactly 5 prints total) ----
            if verbose and step in report_steps:
                print(f"Step {step:4d} | Loss {loss_value:.6f}")

        # ---- Restore best model ----
        with torch.no_grad():
            self.h.copy_(best_h)
            self.J.copy_(self._symmetrize_J(best_J))

        # Compute final marginals
        final_mean, _ = self.model_marginals(max_iter=max_iter, tol=tol, damping=damping)

        if verbose:
            print("Restored best model.")
            print("Final <m> range:", final_mean.min().item(), "–", final_mean.abs().max().item())

        return best_loss
    
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
