import numpy as np
data= np.loadtxt("data4hw.csv", delimiter= ",")
mu= np.mean(data)
sigma= np.std(data)
print (f"Mean = {mu:.2f}, Std Dev = {sigma:.2f}")
def gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (x - mu)**2 / (2 * sigma**2))
import matplotlib.pyplot as plt

# Histogram
plt.hist(data, bins=15, density=True, alpha=0.6, color='skyblue')

# Gaussian curve
x = np.linspace(min(data), max(data), 1000)
pdf = gaussian_pdf(x, mu, sigma)
plt.plot(x, pdf, 'r-', linewidth=2)

plt.title(f"Gaussian Fit (mean={mu:.2f}, std={sigma:.2f})")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()
