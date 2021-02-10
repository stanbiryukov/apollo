# apollo
Highly performant and scalable out-of-the-box gaussian process regression and binomial classification. Built upon GPyTorch, with a familiar sklearn api.

# Examples
## Pattern learning w/ SpectralMixture Kernel
```
from apollo.ml import GP
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

train_x = np.arange(0, 1, .1).reshape(-1,1)
train_y = np.sin(train_x * (2 * np.pi))
ml = GP(kernel=gpytorch.kernels.SpectralMixtureKernel(num_mixtures=9))
ml.fit(train_x, train_y)
X_ = np.linspace(train_x.min()*.75, train_x.max()*5, 1000).reshape(-1,1).astype(np.float32)
hat = ml.predict(X_, sigma=st.norm.ppf(.975))
plt.figure(figsize=(22, 8.5))
plt.scatter(train_x, train_y)
plt.plot(X_, hat[:,0], color='orange')
plt.fill_between(X_.reshape(-1,), hat[:,1], hat[:,2], color='blue', alpha=.5)
```
