# apollo
CUDA accelerated highly performant and scalable out-of-the-box gaussian process regression and Bernoulli classification. Built upon GPyTorch, with a familiar sklearn api.

# Examples
## Pattern learning w/ SpectralMixture Kernel
```python
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
hat = ml.predict(X_, sigma=st.norm.ppf(.975)) # predict w/ 95% UI
plt.figure(figsize=(22, 8.5))
plt.scatter(train_x, train_y)
plt.plot(X_, hat[:,0], color='orange')
plt.fill_between(X_.reshape(-1,), hat[:,1], hat[:,2], color='blue', alpha=.5)
```
## Sparse GP
```python
def func(x):
    return np.sin(x * 2 * np.pi) + 0.4 * np.cos(x * 5 * np.pi) + 0.7 * np.sin(x * 6 * np.pi)

N = 10000
rng = np.random.RandomState(3685)
X = rng.rand(N, 1) * 2 - 1
Y = func(X) + 0.25 * rng.randn(N, 1)

ml = GP(verbose=True, partition_kernel=False, sparse=True, kernel=gpytorch.kernels.MaternKernel(nu=2.5))
ml.fit(X, Y)

X_ = np.linspace(np.min(X), np.max(X)*1.5, 1000).reshape(-1,1)
hat = ml.predict(X_, sigma=1.96)
plt.plot(X, Y, "x", color='blue', alpha=.01)
plt.plot(X_, hat[:,0], c="k", alpha=.5, color='purple')
plt.fill_between(X_.reshape(-1,), hat[:,1], hat[:,2], color='purple', alpha=.5)
```
## Performant ML solver
```python
from sklearn import datasets
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

X, y = datasets.load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3249)

ml = GP(l2_reg=True)
ml.fit(X=X_train, y=y_train)

metrics.mean_squared_error(y_test, ml.predict(X=X_test), squared=True)
```
## Binomial classification
```python
from sklearn.datasets import make_moons
from sklearn.preprocessing import FunctionTransformer
import matplotlib

# Create dataset 
X, y = make_moons(noise=0.3, n_samples=100, random_state=0)

# Create evalutaion grid
h = 0.05
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
X_eval = np.vstack((xx.reshape(-1), 
                    yy.reshape(-1))).T

# binomial likelihood w/ anisotropic Matern kernel; keep y in 0-1 space.
ml = GP(problem='binomial', y_scaler=FunctionTransformer(), kernel=gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))
ml.fit(X, y)

# predict
Z_skl = ml.predict(np.vstack((xx.ravel(), yy.ravel())).T).reshape(xx.shape)

normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
f, ax = plt.subplots(figsize=(10, 10))
im = ax.contourf(xx, yy, Z_skl, norm=normalize, levels=16)
im2 = ax.scatter(X[:,0], X[:,1], c=y, norm=normalize)
f.colorbar(im)

```
