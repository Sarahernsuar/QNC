import numpy as np
import matplotlib.pyplot as plt

# ---- Parameters ----
# alpha is the standard false positive rate 
#80% chance to determine if the effect is real
alpha = 0.05      # significance level
power = 0.80      # desired power
effect_sizes = np.round(np.linspace(0.1, 1.2, 12), 2)  # Cohen's d values

# ---- Z values ----
# Determine the standard deviations away from the mean from the normal distribution
# Instead of scipy, we can use numpy.percentile equivalents
# Z_{1-alpha/2} ~ 1.96 for alpha=0.05
z_alpha = 1.96
# Z for 80% power (1-beta), beta=0.2 → z ~ 0.84
z_power = 0.84

# ---- Compute required n per group ----
required_n = ((z_alpha + z_power)**2) / (effect_sizes**2)
required_n = np.ceil(required_n).astype(int)  # round up to whole sessions

# ---- Print table ----
print("Effect size (d)  ->  Required sessions per group (80% power, alpha=0.05):")
for d, n in zip(effect_sizes, required_n):
    print(f"{d:>4.2f} -> {n:>4d}")

# ---- Plot ----
plt.plot(effect_sizes, required_n, marker="o")
plt.xlabel("Effect size (Cohen's d)")
plt.ylabel("Required sessions per group")
plt.title("Post-hoc power (z-test approximation)")
plt.grid(True)
plt.tight_layout()
plt.savefig("power_vs_effect_size_simple.png", dpi=150)
plt.show()