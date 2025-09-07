from math import comb
#Exercise #1 
# Parameters
n = 10   # total quanta
p = 0.2  # probability of release

# Compute probabilities for 0 through n quanta released
probs = []
for k in range(n+1):
    prob = comb(n, k) * (p**k) * ((1-p)**(n-k))
    probs.append(prob)
    print(f"P({k} quanta) = {prob:.5f}")

# Check that probabilities sum to ~1
print("Sum of probabilities:", sum(probs))
#P(0 quanta) = 0.10737
#P(1 quanta) = 0.26844
#P(2 quanta) = 0.30199
#P(3 quanta) = 0.20133
#P(4 quanta) = 0.08808
#P(5 quanta) = 0.02642
#P(6 quanta) = 0.00551
#P(7 quanta) = 0.00079
#P(8 quanta) = 0.00007
#P(9 quanta) = 0.00000
#P(10 quanta) = 0.00000
#Exercise #2
from math import comb

# Parameters
n = 14   # number of available quanta
k = 8    # observed releases

# Probabilities to test
p_values = [i/10 for i in range(1, 11)]  # 0.1, 0.2, ..., 1.0

results = {}
for p in p_values:
    prob = comb(n, k) * (p**k) * ((1-p)**(n-k))
    results[p] = prob
    print(f"P(X=8 | p={p:.1f}) = {prob:.6f}")

# Find which p is most likely
best_p = max(results, key=results.get)
print(f"\nMost probable release probability given data: p = {best_p:.1f}")
#If the true probability of release for 8 quanta was p.1, there is a 0.000016 chance of that occuring. For 8 quanta the most likely scenario is p.5 or p 0.6 which yield 0.18 and 0.20 chance respectively
#Exercise #3-#5 Unfortinately I could not figure out how to go about these problems
