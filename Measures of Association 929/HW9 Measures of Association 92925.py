import math, random
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
# Age (x) and Wing Length (y)
x = [3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17]
y = [1.4, 1.5, 2.2, 2.4, 3.1, 3.2, 3.2, 3.9, 4.1, 4.7, 4.5, 5.2, 5.0]

n = len(x)

# -----------------------------
# Helper functions
# -----------------------------

def mean(a):
    return sum(a) / len(a)

def Sxx(x):
    mx = mean(x)
    return sum((xi - mx)**2 for xi in x)

def Syy(y):
    my = mean(y)
    return sum((yi - my)**2 for yi in y)

def Sxy(x, y):
    mx, my = mean(x), mean(y)
    return sum((xi - mx)*(yi - my) for xi, yi in zip(x, y))

def linear_regression(x, y):
    """Return slope (b1) and intercept (b0) for y ~ b0 + b1*x"""
    b1 = Sxy(x, y) / Sxx(x)
    b0 = mean(y) - b1 * mean(x)
    return b1, b0

def residuals(x, y, b1, b0):
    return [yi - (b0 + b1*xi) for xi, yi in zip(x, y)]

def mse_and_se_slope(x, y, b1, b0):
    """Return MSE (sigma^2_hat) and standard error of slope."""
    r = residuals(x, y, b1, b0)
    sse = sum(ri**2 for ri in r)
    df = len(x) - 2
    mse = sse / df
    se_b1 = math.sqrt(mse / Sxx(x))
    return mse, se_b1

def pearson_r(x, y):
    return Sxy(x, y) / math.sqrt(Sxx(x) * Syy(y))

def r_squared(x, y, b1, b0):
    my = mean(y)
    sst = sum((yi - my)**2 for yi in y)
    sse = sum((yi - (b0 + b1*xi))**2 for xi, yi in zip(x, y))
    return 1 - sse/sst

def bootstrap_ci_slope(x, y, B=5000, alpha=0.05, seed=123):
    """Nonparametric bootstrap CI for slope by resampling (x,y) pairs with replacement."""
    random.seed(seed)
    n = len(x)
    slopes = []
    indices = list(range(n))
    for _ in range(B):
        samp_idx = [random.choice(indices) for __ in range(n)]
        xb = [x[i] for i in samp_idx]
        yb = [y[i] for i in samp_idx]
        # Need Sxx>0, which will be true unless all xb are identical (very unlikely but guard it)
        if Sxx(xb) == 0:
            continue
        b1b, _ = linear_regression(xb, yb)
        slopes.append(b1b)
    slopes.sort()
    lo_idx = int((alpha/2) * len(slopes))
    hi_idx = int((1 - alpha/2) * len(slopes)) - 1
    return slopes[lo_idx], slopes[hi_idx], slopes

def permutation_test_no_relationship(x, y, B=10000, seed=123):
    """
    Permutation test for H0: no relationship between x and y (slope = 0).
    Test statistic: absolute value of Pearson r.
    Returns (p_value, observed_statistic).
    """
    random.seed(seed)
    obs_r = abs(pearson_r(x, y))
    y_perm = y[:]
    count = 0
    for _ in range(B):
        random.shuffle(y_perm)
        stat = abs(pearson_r(x, y_perm))
        if stat >= obs_r - 1e-15:
            count += 1
    p_val = count / B
    return p_val, obs_r

# -----------------------------
# Exercise 1 — Plot the relationship (scatter plot)
# -----------------------------
plt.figure()
plt.title("Exercise 1: Age vs Wing Length")
plt.xlabel("Age")
plt.ylabel("Wing Length")
plt.scatter(x, y)
plt.tight_layout()
plt.savefig("ex1_scatter.png", dpi=150)
plt.show()

# -----------------------------
# Exercise 2 — Calculate and plot the regression line
# -----------------------------
b1, b0 = linear_regression(x, y)

# line for plotting across the data range
x_min, x_max = min(x), max(x)
x_line = [x_min + t*(x_max - x_min)/200 for t in range(201)]
y_line = [b0 + b1*xi for xi in x_line]

plt.figure()
plt.title("Exercise 2: Regression Line")
plt.xlabel("Age")
plt.ylabel("Wing Length")
plt.scatter(x, y, label="Data")
plt.plot(x_line, y_line, label=f"y = {b0:.3f} + {b1:.3f}·x")
plt.legend()
plt.tight_layout()
plt.savefig("ex2_regression.png", dpi=150)
plt.show()

# -----------------------------
# Exercise 3 — Hypothesis test: Can you reject H0 (no relationship)?
# Implemented via a permutation test on |r|.
# -----------------------------
p_val, obs_stat = permutation_test_no_relationship(x, y, B=10000, seed=42)

# -----------------------------
# Exercise 4 — 95% Confidence Interval for the slope (bootstrap) and plot
# We'll plot the lower/upper CI slope lines anchored at (x̄, ȳ) to visualize slope uncertainty.
# -----------------------------
ci_lo, ci_hi, slope_samples = bootstrap_ci_slope(x, y, B=5000, alpha=0.05, seed=7)
mx, my = mean(x), mean(y)
# Define intercepts so all lines pass through the point (x̄,ȳ)
b0_lo, b0_hi = my - ci_lo*mx, my - ci_hi*mx

y_line_lo = [b0_lo + ci_lo*xi for xi in x_line]
y_line_hi = [b0_hi + ci_hi*xi for xi in x_line]

plt.figure()
plt.title("Exercise 4: 95% CI for Slope (Bootstrap)")
plt.xlabel("Age")
plt.ylabel("Wing Length")
plt.scatter(x, y, label="Data")
plt.plot(x_line, y_line, label=f"Fit: y = {b0:.3f} + {b1:.3f}·x")
plt.plot(x_line, y_line_lo, label=f"Lower slope {ci_lo:.3f}")
plt.plot(x_line, y_line_hi, label=f"Upper slope {ci_hi:.3f}")
plt.legend()
plt.tight_layout()
plt.savefig("ex4_slope_ci.png", dpi=150)
plt.show()

# -----------------------------
# Exercise 5 — R^2 (coefficient of determination)
# -----------------------------
def r_squared(x, y, b1, b0):
    my = mean(y)
    sst = sum((yi - my)**2 for yi in y)
    sse = sum((yi - (b0 + b1*xi))**2 for xi, yi in zip(x, y))
    return 1 - sse/sst

R2 = r_squared(x, y, b1, b0)

# -----------------------------
# Exercise 6 — Pearson's r
# -----------------------------
r_val = pearson_r(x, y)

# -----------------------------
# Exercise 7 — Add noise and see how regression changes
# We'll add zero-mean Gaussian noise with sigma = 0.3, then refit.
# -----------------------------
random.seed(1234)
sigma = 0.3
y_noisy = [yi + random.gauss(0.0, sigma) for yi in y]
b1_n, b0_n = linear_regression(x, y_noisy)

y_line_n = [b0_n + b1_n*xi for xi in x_line]

plt.figure()
plt.title("Exercise 7: Regression After Adding Noise (σ=0.3)")
plt.xlabel("Age")
plt.ylabel("Wing Length")
plt.scatter(x, y_noisy, label="Noisy Data")
plt.plot(x_line, y_line_n, label=f"Noisy fit: y = {b0_n:.3f} + {b1_n:.3f}·x")
plt.legend()
plt.tight_layout()
plt.savefig("ex7_noisy_regression.png", dpi=150)
plt.show()

# -----------------------------
# Print all numeric results clearly
# -----------------------------
print("========== RESULTS ==========")
print("Exercise 2 — Regression coefficients")
print(f"  Intercept (b0): {b0:.6f}")
print(f"  Slope (b1)    : {b1:.6f}")

mse, se_b1 = mse_and_se_slope(x, y, b1, b0)
print("\nExercise 3 — Hypothesis test (Permutation)")
print("  H0: no relationship (slope = 0); test statistic = |r|")
print(f"  Observed |r| : {obs_stat:.6f}")
print(f"  p-value     : {p_val:.6f} (two-sided, permutation)")

print("\nExercise 4 — 95% Bootstrap CI for slope")
print(f"  95% CI for slope: [{ci_lo:.6f}, {ci_hi:.6f}]")
print(f"  (Std. Error of slope from residual formula, FYI): {se_b1:.6f}")

print("\nExercise 5 — Coefficient of determination (R^2)")
print(f"  R^2: {R2:.6f}")

print("\nExercise 6 — Pearson's r")
print(f"  r: {r_val:.6f}")

print("\nExercise 7 — Noisy regression (σ=0.3)")
print(f"  Intercept (noisy): {b0_n:.6f}")
print(f"  Slope (noisy)    : {b1_n:.6f}")