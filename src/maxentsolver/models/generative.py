import torch
import torch.nn as nn

class GenMaxEnt(nn.Module):
    def __init__(self, h, J, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        h = torch.as_tensor(h, dtype=torch.float32)
        J = torch.as_tensor(J, dtype=torch.float32)
        n = h.shape[0]
        assert J.shape == (n, n), "J must be of shape (n, n)"
        assert torch.allclose(J, J.t()), "J must be symmetric"
        assert torch.all(torch.diag(J) == 0).item(), "J must have zero diagonal"
        
        # Register as buffers so they live on the module's device and aren't treated as parameters
        self.register_buffer('h', h.to(self.device))
        self.register_buffer('J', J.to(self.device))
        self.n = n

    def forward(self, s):
        s = s.to(self.h.device)  # ensure same device
        # Energy: E(s) = - h^T s - 1/2 s^T J s
        h_term = -torch.sum(self.h * s, dim=1)
        J_term = -0.5 * torch.sum((s @ self.J) * s, dim=1)
        return h_term + J_term
    
    def generate(self, num_samples, num_sweeps=10_000, sequential=True, random_site_order=True):
        """
        Gibbs / heat-bath sampler.
        If sequential=True, updates spins one site at a time (correct detailed balance).
        If sequential=False, updates all spins simultaneously (not recommended).
        """
        device = self.h.device
        samples = (torch.rand(num_samples, self.n, device=device) > 0.5).float() * 2 - 1  # {-1,+1}
        
        J = self.J
        h = self.h
        
        if not sequential:
            # synchronous updates (fast but may not converge to correct Gibbs distribution)
            for _ in range(num_sweeps):
                fields = samples @ J.t() + h.unsqueeze(0)
                prob = torch.sigmoid(2 * fields)
                samples = 2 * torch.bernoulli(prob) - 1
            return samples
        
        # sequential updates (recommended)
        for sweep in range(num_sweeps):
            if random_site_order:
                site_order = torch.randperm(self.n, device=device)
            else:
                site_order = torch.arange(self.n, device=device)
            for i in site_order:
                # compute field at site i for all samples
                # field = h[i] + sum_j J[i, j] * samples[:, j]
                field_i = h[i] + samples @ J[:, i]
                prob = torch.sigmoid(2 * field_i)
                samples[:, i] = 2 * torch.bernoulli(prob) - 1
        return samples