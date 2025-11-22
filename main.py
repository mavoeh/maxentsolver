import torch
from src.maxentsolver import MaxEnt, plot_maxent_results


# Now just use the unified interface
n_neurons = 80
data = (torch.randn(10000, n_neurons) > 0).float()  # fake data

models = {}
for model_type in ["mcmc", "meanfield"]:
    model = MaxEnt(n=n_neurons, method=model_type)
    model.fit(data, steps=2000 if model_type == "mcmc" else 500)
    mean, cov_flat = model.model_marginals()
    J = model.interaction_matrix()
    print(f"Method: {model.method.upper()} | Mean firing rate: {mean.mean():.3f}")
    models[model_type] = model


# Plot results (your function from before)
plot_maxent_results(data, models.get("mcmc"), title="MCMC MaxEnt Model")
plot_maxent_results(data, models.get("meanfield"), title="Mean-Field MaxEnt Model")
