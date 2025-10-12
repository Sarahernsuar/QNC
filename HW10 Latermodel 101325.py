import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize  #searched for the best parameters
TWOPI = 2.0 * np.pi
# Loading Data 
# RT (reaction times) eps (avoid dividing by zero) V (reciprocal reaction. times 1/RT ) np.max (ensures no divisions smaller than eps)
def normal_logpdf(x, mu, sigma):
    z = (x - mu) / sigma
    return -0.5*np.log(TWOPI) - np.log(sigma) - 0.5*z*z
def normal_ppf(p):
    p = np.asarray(p, dtype=float)
    eps = np.finfo(float).eps
    p = np.clip(p, eps, 1 - eps)

    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]

    plow, phigh = 0.02425, 1 - 0.02425
    x = np.empty_like(p)
    lo = p < plow
    ce = (p >= plow) & (p <= phigh)
    hi = p > phigh
    if np.any(lo):
        q = np.sqrt(-2*np.log(p[lo]))
        x[lo] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if np.any(ce):
        q = p[ce] - 0.5
        r = q*q
        x[ce] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
                 (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    if np.any(hi):
        q = np.sqrt(-2*np.log(1 - p[hi]))
        x[hi] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                  ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    return x

RTs = np.array([0.31, 0.28, 0.35, 0.42, 0.37, 0.30, 0.33, 0.36, 0.29, 0.40])
eps = 1e-9
V = 1.0 / np.maximum(RTs, eps) #1/RT



#define objectives: Neg Log Likelihood 
# params ( LATER parameters) normal log of (LLL of each data point and sume them to have the total LL) 
# For these parameters, how likely is my data if 1/RT is normally distributed. Best parameters minimize the negative log likelihood
def later_nll(params):
    muR, deltaS = params
    if muR <= 0 or deltaS <= 0 or not np.isfinite(muR+deltaS):
        return 1e12
    mu = muR / deltaS
    sigma = 1.0 / deltaS
    if not np.isfinite(sigma) or sigma <= 0:
        return 1e12
    ll = normal_logpdf(V, mu, sigma).sum()
    return -ll 

# Data driven initial guesses 
# mu hat (mean 1/RT) sigma hat (std 1/RT)
mu_hat = V.mean()
sigma_hat = V.std()
initial_guess = np.array([mu_hat / max(sigma_hat, eps), 1.0 / max(sigma_hat, eps)])  # [muR, deltaS]

#Fit with bounds 
#bounds (paramters have to stay positive and within a wide but reasonable range) res.x (fitted) res.fun (final minimized NLL) 
bounds = [(1e-3, 1e3), (1e-3, 1e3)]
res = minimize(later_nll, x0=initial_guess, bounds=bounds, method='L-BFGS-B')
muR, deltaS = res.x
nllk = res.fun

# Evaluate the fit 
# converting fitted LATER parameters back to normal distribution parameters. It should similar to the emirical mean and standard deviation AIC/BIC (better is smaller ratio models/conditions)
mu_fit = muR / deltaS      # model mean in 1/RT space
sigma_fit = 1.0 / deltaS   # model std  in 1/RT space
n = RTs.size
k = 2
AIC = 2*k + 2*nllk
BIC = k*np.log(n) + 2*nllk

print("\n--- LATER fit summary (no scipy.stats) ---")
print(f"success            = {res.success} ({res.message})")
print(f"muR                = {muR:.6f}")
print(f"deltaS             = {deltaS:.6f}")
print(f"mean(1/RT) data    = {mu_hat:.6f}")
print(f"std(1/RT)  data    = {sigma_hat:.6f}")
print(f"mean(1/RT) model   = {mu_fit:.6f}")
print(f"std(1/RT)  model   = {sigma_fit:.6f}")
print(f"NLL                = {nllk:.6f}")
print(f"AIC                = {AIC:.6f}")
print(f"BIC                = {BIC:.6f}")

# Visual check 
# Histogram + fitted Normal pdf
plt.figure(figsize=(6,4))
counts, bins, _ = plt.hist(V, bins='auto', density=True, alpha=0.6, edgecolor='k')
x = np.linspace(V.min(), V.max(), 400)
pdf = (1.0/(sigma_fit*np.sqrt(TWOPI))) * np.exp(-0.5*((x-mu_fit)/sigma_fit)**2)
plt.plot(x, pdf, lw=2)
plt.xlabel('1/RT'); plt.ylabel('Density'); plt.title('LATER fit in 1/RT space'); plt.grid(alpha=0.3)

#Plot empirical distribution of 1/RT and overlayes the fitted normal curve 
# Q–Q compares empirical vs model quantiles
# If points near the diagnoal the nomral assumpation is resonable 
q = np.linspace(0.01, 0.99, 50)          # probabilities
emp_q = np.quantile(V, q)                # empirical quantiles of 1/RT
mod_q = mu_fit + sigma_fit * normal_ppf(q)  # model's Normal quantiles

plt.figure(figsize=(5,5))
plt.plot(mod_q, emp_q, 'o', ms=5)
plt.plot(mod_q, mod_q, '-', lw=1.5)     # reference y=x line
plt.xlabel('Model quantiles (Normal)'); plt.ylabel('Empirical quantiles (1/RT)')
plt.title('Q–Q check for 1/RT ~ Normal'); plt.axis('equal'); plt.grid(alpha=0.3)
plt.show()
