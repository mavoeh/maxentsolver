import torch
import matplotlib.pyplot as plt
from src.maxentsolver import GenMaxEnt, MaxEnt, plot_maxent_results


n = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

h = torch.randn(n) * 0.2
J = torch.randn(n, n) * 0.2
J = (J + J.t()) / 2
J.fill_diagonal_(0)
true_interaction_matrix = torch.diag(h) + J

print("Generating synthetic data...")
gen = GenMaxEnt(h, J, device=device)
data = gen.generate(num_samples=100_000, num_sweeps=100_000)


print("Fitting MaxEnt model using mean-field approximation...")
model = MaxEnt(n=n, method="meanfield", device=device)
model.fit(data, lr=1e-2, steps=2000, total_reports=2000)
interaction_matrix = model.interaction_matrix()


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(true_interaction_matrix.numpy(), cmap='bwr', vmin=-0.5, vmax=0.5)
axs[0].set_title("True Interaction Matrix")
axs[1].imshow(interaction_matrix, cmap='bwr', vmin=-0.5, vmax=0.5)
axs[1].set_title("Inferred Interaction Matrix")

# Plot results (your function from before)
plot_maxent_results(data, model, title="Mean-Field MaxEnt Model")
plt.show()