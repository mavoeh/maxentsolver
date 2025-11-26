import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

def plot_maxent_results(data, model, title=None, marginals_kwargs={}, verbose=False):
    """
    Clean, publication-ready figure per model:
    • Firing rates (emp vs model)
    • Scatter of single-neuron marginals (emp vs model)
    • Scatter of pairwise marginals (emp vs model)
    • Summary box on the right
    """
    data = torch.as_tensor(data, dtype=torch.float)
    device = next(model.parameters()).device
    data = data.to(device)
    n = data.shape[1]

    with torch.no_grad():
        emp_mean, emp_corr_flat = model.get_empirical_marginals(data)
        model_mean, model_corr_flat = model.model_marginals(verbose=verbose, **marginals_kwargs)

    # To numpy
    emp_mean = emp_mean.cpu().numpy()
    model_mean = model_mean.cpu().numpy()
    emp_corr = emp_corr_flat.cpu().numpy()
    model_corr = model_corr_flat.cpu().numpy()

    # R² scores
    r2_mean = 1 - np.sum((emp_mean - model_mean)**2) / (np.sum((emp_mean - emp_mean.mean())**2) + 1e-12)
    r2_corr = 1 - np.sum((emp_corr - model_corr)**2) / (np.sum((emp_corr - emp_corr.mean())**2) + 1e-12)

    # === Plotting layout ===
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)


    ax1 = fig.add_subplot(1, 3, 1)  # dummy to initialize
    lim_mean = max(max(abs(emp_mean)), max(abs(model_mean))) * 1.1
    ax1.scatter(emp_mean, model_mean, s=20, c='#1f77b4', alpha=0.75, edgecolors='none')
    ax1.plot([-lim_mean, lim_mean], [-lim_mean, lim_mean], '--', color='gray', lw=1.5)
    ax1.set_xlim(-lim_mean, lim_mean)
    ax1.set_ylim(-lim_mean, lim_mean)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title("Mean marginals")
    ax1.set_xlabel("Empirical ⟨sᵢ⟩")
    ax1.set_ylabel("Model ⟨sᵢ⟩")
    ax1.grid(True, alpha=0.3)

    # --- 3. Pairwise correlations (bottom-left)
    ax2 = fig.add_subplot(1, 3, 2)
    lim = max((max(abs(emp_corr)), max(abs(model_corr)))) * 1.1
    ax2.scatter(emp_corr, model_corr, c='#1f77b4', s=14, alpha=0.75, edgecolors='none')
    ax2.plot([-lim, lim], [-lim, lim], '--', color='gray', lw=1.5)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel("Empirical ⟨sᵢsⱼ⟩ − ⟨sᵢ⟩⟨sⱼ⟩")
    ax2.set_ylabel("Model ⟨sᵢsⱼ⟩ − ⟨sᵢ⟩⟨sⱼ⟩")
    ax2.set_title("Pairwise marginals")
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.95, f"R² = {r2_corr:.4f}", transform=ax2.transAxes,
             fontsize=11, va='top',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

    # --- 4. Summary box (right column spanning all rows)
    axsum = fig.add_subplot(1, 3, 3)
    axsum.axis('off')

    summary = f"""
MAXIMUM ENTROPY MODEL FIT

Method              │ {model.method.upper():<18}
# of Nodes (n)      │ {n:<18}
Samples             │ {len(data):<18}
Training device     │ {device.__str__():<18}

Goodness-of-fit (R²)
┌────────────────────────────────────────┐
│ Single-neuron marginals      → {r2_mean:6.4f} │
│ Pairwise marginals           → {r2_corr:6.4f} │
└────────────────────────────────────────┘
"""
    axsum.text(
        0.5, 0.5, summary,
        ha='center', va='center',
        fontsize=13, fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=1.2",
                  facecolor="#f8f8f8",
                  edgecolor="#3339", linewidth=1.5)
    )

    # Title
    if title is None:
        title = f"MaxEnt • {model.method.upper()} • n = {n} neurons"
    fig.suptitle(title, fontsize=18, y=0.98, weight='bold')
    
    print(f"{model.method.upper():10} → R² mean: {r2_mean:.4f} | R² pairwise: {r2_corr:.4f}")
