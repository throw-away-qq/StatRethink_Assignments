import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
np.random.seed(100)

## steps
## initialize the search grid
## initalize prior (all are equal likely)
## using given evidence use probability desnity function to get likelihood
## get posterior using likelihood and prior
## normalizen posterior and use it to sample another prior


## flat prior
p_grid = np.arange(0.0,1.0, 0.001)
# both prob_p and p_grid will give prior
prob_p = np.ones(1000)

plt.plot(prob_p)
plt.show()

likelihood = binom.pmf(8, 15, p_grid)
posterior= likelihood*prob_p

plt.plot(posterior)
plt.show()

posterior = posterior/np.sum(posterior)

sample = np.random.choice(p_grid, size = 10000, p = posterior)

## mean probability
np.mean(sample)

## 99 percentile and 1 percentile
(0.261, 0.791)
np.quantile(sample, 0.01)
np.quantile(sample, 0.99)


## step function prior
p_grid = np.arange(0.0,1.0, 0.001)
# both prob_p and p_grid will give prior
prob_p = np.concatenate([np.zeros(500),\
    np.ones(500)])

plt.plot(prob_p)
plt.show()

likelihood = binom.pmf(8, 15, p_grid)
posterior= likelihood*prob_p

plt.plot(posterior)
plt.show()

posterior = posterior/np.sum(posterior)

sample = np.random.choice(p_grid, size = 10000, p = posterior)

## mean probability 0.61
np.mean(sample)

## 99 percentile and 1 percentile
# (0.5, 0.81)
np.quantile(sample, 0.01)
np.quantile(sample, 0.99)




# ##3 iterative
# ## flat prior
# p_grid = np.arange(0.0,1.0, 0.0001)
# # both prob_p and p_grid will give prior
# prob_p = np.ones(10000)

# plt.plot(prob_p)
# plt.show()
# for _ in range(5):
#     likelihood = binom.pmf(8, 15, p_grid)
#     posterior= likelihood*prob_p

#     plt.plot(posterior)
#     plt.show()

#     posterior = posterior/np.sum(posterior)

#     sample = np.random.choice(p_grid, size = 10000, p = posterior)

#     ## mean probability
#     print(np.mean(sample),np.quantile(sample, 0.01) , np.quantile(sample, 0.99))

#     ## 99 percentile and 1 percentile
#     # (0.261, 0.791)
#     prob_p = sample
