import copy
import gc
import time
from functools import partial

import gpytorch
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchLBFGS import FullBatchLBFGS


def set_seed(seed):
    import random

    import numpy as np
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def TorchRandom(X, n_inducing, seed):
    import random

    random.seed(seed)
    indice = random.sample(range(X.shape[0]), k=np.min([n_inducing, X.shape[0]]))
    indice = torch.as_tensor(indice)
    sampled_values = X[indice]
    return sampled_values


def SGPRInducing(X, n_inducing):
    from scipy.cluster.vq import kmeans

    Xu, _ = kmeans(X.cpu().numpy(), np.min([n_inducing, X.shape[0]]))
    return torch.as_tensor(Xu[:, 0 : X.shape[-1]]).type(torch.FloatTensor)


class NoneStep:
    """
    Dummy scheduler in case a real one isn't used.
    """

    def __init__(self):
        step = None

    def step(self):
        return None


class ExpMAStoppingCriterion:
    r"""Exponential moving average stopping criterion from BoTorch.
    # https://github.com/pytorch/botorch/blob/master/botorch/optim/stopping.py
    Computes an exponentially weighted moving average over window length `n_window`
    and checks whether the relative decrease in this moving average between steps
    is less than a provided tolerance level. That is, in iteration `i`, it computes
        v[i,j] := fvals[i - n_window + j] * w[j]
    for all `j = 0, ..., n_window`, where `w[j] = exp(-eta * (1 - j / n_window))`.
    Letting `ma[i] := sum_j(v[i,j])`, the criterion evaluates to `True` whenever
        (ma[i-1] - ma[i]) / abs(ma[i-1]) < rel_tol (if minimize=True)
        (ma[i] - ma[i-1]) / abs(ma[i-1]) < rel_tol (if minimize=False)
    """

    def __init__(
        self,
        maxiter: int = 10000,
        minimize: bool = True,
        n_window: int = 10,
        eta: float = 1.0,
        rel_tol: float = 1e-5,
    ) -> None:
        r"""Exponential moving average stopping criterion.
        Args:
            maxiter: Maximum number of iterations.
            minimize: If True, assume minimization.
            n_window: The size of the exponential moving average window.
            eta: The exponential decay factor in the weights.
            rel_tol: Relative tolerance for termination.
        """
        self.maxiter = maxiter
        self.minimize = minimize
        self.n_window = n_window
        self.rel_tol = rel_tol
        self.iter = 0
        weights = torch.exp(torch.linspace(-eta, 0, self.n_window))
        self.weights = weights / weights.sum()
        self._prev_fvals = None

    def evaluate(self, fvals: torch.as_tensor) -> bool:
        r"""Evaluate the stopping criterion.
        Args:
            fvals: tensor containing function values for the current iteration. If
                `fvals` contains more than one element, then the stopping criterion is
                evaluated element-wise and True is returned if the stopping criterion is
                true for all elements.
        TODO: add support for utilizing gradient information
        Returns:
            Stopping indicator (if True, stop the optimziation).
        """
        self.iter += 1
        if self.iter == self.maxiter:
            return True

        if self._prev_fvals is None:
            self._prev_fvals = fvals.unsqueeze(0)
        else:
            self._prev_fvals = torch.cat(
                [self._prev_fvals[-self.n_window :], fvals.unsqueeze(0)]
            )

        if self._prev_fvals.size(0) < self.n_window + 1:
            return False

        weights = self.weights
        weights = weights.to(fvals)
        if self._prev_fvals.ndim > 1:
            weights = weights.unsqueeze(-1)

        # TODO: Update the exp moving average efficiently
        prev_ma = (self._prev_fvals[:-1] * weights).sum(dim=0)
        ma = (self._prev_fvals[1:] * weights).sum(dim=0)
        # TODO: Handle approx. zero losses (normalize by min/max loss range)
        rel_delta = (prev_ma - ma) / prev_ma.abs()

        if not self.minimize:
            rel_delta = -rel_delta
        if torch.max(rel_delta) < self.rel_tol:
            return True

        return False


class cGPR(gpytorch.models.AbstractVariationalGP):
    def __init__(
        self,
        kernel,
        mean_module,
        train_x,
        train_y,
        likelihood,
        seed,
        feature_extractor=None,
        add_feature=False,
        fit_intercept=True,
    ):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            train_x.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, train_x, variational_distribution
        )
        super(cGPR, self).__init__(variational_strategy)
        self.likelihood = likelihood
        self.base_covar_module = kernel
        self.feature_extractor = feature_extractor
        self.add_feature = add_feature
        set_seed(seed)
        if mean_module in [gpytorch.means.LinearMean]:
            if fit_intercept in [1]:
                bias = True
            else:
                bias = None
            if (self.feature_extractor is not None) and (self.add_feature in [1]):
                fin_dims = train_x.shape[-1] + self.feature_extractor.num_features
                self.mean_module = mean_module(input_size=fin_dims, bias=bias)
            elif (self.feature_extractor is not None) and (self.add_feature in [0]):
                fin_dims = self.feature_extractor.num_features
                self.mean_module = mean_module(input_size=fin_dims, bias=bias)
            else:
                self.mean_module = mean_module(input_size=train_x.shape[-1], bias=bias)
        else:
            self.mean_module = mean_module()
        self.covar_module = self.base_covar_module

    def forward(self, x):
        if (self.feature_extractor is not None) and (self.add_feature in [1]):
            xf = self.feature_extractor(x)
            x = torch.cat([x, xf], dim=1)
        elif (self.feature_extractor is not None) and (self.add_feature in [0]):
            x = self.feature_extractor(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPR(gpytorch.models.ExactGP):
    def __init__(
        self,
        kernel,
        mean_module,
        train_x,
        train_y,
        likelihood,
        seed,
        sparse,
        n_inducing,
        device,
        feature_extractor=None,
        add_feature=False,
        fit_intercept=True,
    ):
        super(GPR, self).__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        # if "pectral" in kernel.__class__.__name__:
        #    self.base_covar_module = kernel  # don't scale Spectral Kernel.
        # else:
        # self.base_covar_module = gpytorch.kernels.ScaleKernel(kernel)
        self.base_covar_module = kernel
        self.feature_extractor = feature_extractor
        self.add_feature = add_feature
        self.device = device
        set_seed(seed)
        if mean_module in [gpytorch.means.LinearMean]:

            if fit_intercept in [1]:
                bias = True
            else:
                bias = None
            if (self.feature_extractor is not None) and (self.add_feature in [1]):
                fin_dims = train_x.shape[-1] + self.feature_extractor.num_features
                self.mean_module = mean_module(input_size=fin_dims, bias=bias)
            elif (self.feature_extractor is not None) and (self.add_feature in [0]):
                fin_dims = self.feature_extractor.num_features
                self.mean_module = mean_module(input_size=fin_dims, bias=bias)
            else:
                self.mean_module = mean_module(input_size=train_x.shape[-1], bias=bias)

        else:
            self.mean_module = mean_module()

        if "pectral" in kernel.__class__.__name__:
            try:
                # only for specific dims of data.
                self.base_covar_module.initialize_from_data(
                    train_x.to("cpu"), train_y.to("cpu")
                )
            except:
                try:
                    self.base_covar_module.initialize_from_data(train_x, train_y)
                except:
                    print("Initializing from empspect")
                    try:
                        self.base_covar_module.initialize_from_data_empspect(
                            train_x, train_y
                        )
                    except:
                        pass

        if (sparse in [1]) and (self.feature_extractor is None):
            self.covar_module = gpytorch.kernels.InducingPointKernel(
                self.base_covar_module,
                inducing_points=TorchRandom(train_x, n_inducing, seed).to(self.device),
                likelihood=self.likelihood,
            )
        elif (sparse in [1]) and (self.feature_extractor is not None):
            srows = np.min([train_y.shape[0], n_inducing])
            self.covar_module = gpytorch.kernels.InducingPointKernel(
                self.base_covar_module,
                inducing_points=train_x[:srows, -1]
                .repeat(fin_dims, 1)
                .reshape(-1, fin_dims),
                likelihood=self.likelihood,
            )
        else:
            self.covar_module = self.base_covar_module

    def forward(self, x):
        if (self.feature_extractor is not None) and (self.add_feature in [1]):
            xf = self.feature_extractor(x)
            x = torch.cat([x, xf], dim=1)
        elif (self.feature_extractor is not None) and (self.add_feature in [0]):
            x = self.feature_extractor(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP(BaseEstimator):
    """
    Scikit-learn friendly modularized GPyTorch wrapped regressor and classifier.
    """

    def __init__(
        self,
        x_scaler=StandardScaler(),
        y_scaler=StandardScaler(),
        random_state=3892,
        kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5)),
        mean_module=gpytorch.means.LinearMean,
        base_optimizer=partial(
            FullBatchLBFGS, lr=1, line_search="Wolfe", history_size=100
        ),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        scheduler=None,
        verbose=False,
        l2_reg=False,
        l1_reg=False,
        max_iter=1000,
        feature_extractor=None,
        sparse=False,
        alpha=None,
        add_feature=False,
        fit_intercept=False,
        problem="regression",
        partition_kernel=True,
        n_inducing=1024,
        linear_init=torch.nn.init.kaiming_normal_,
        early_stopping=True,
        learn_additional_noise=True,

    ):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.random_state = random_state
        self.kernel = kernel
        self.mean_module = mean_module
        self.base_optimizer = base_optimizer
        self.device = device
        self.scheduler = scheduler
        self.verbose = verbose
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.max_iter = max_iter
        self.feature_extractor = feature_extractor
        self.sparse = sparse
        self.alpha = alpha
        self.add_feature = add_feature
        self.fit_intercept = fit_intercept
        self.problem = problem
        self.partition_kernel = partition_kernel
        self.n_inducing = n_inducing
        self.linear_init = linear_init
        self.early_stopping = early_stopping
        self.learn_additional_noise = learn_additional_noise

    def _to_tensor(self, tensor, dtype=torch.FloatTensor):
        return torch.as_tensor(tensor).to(self.device)

    def _setfit(self, random_state, X, y):
        set_seed(random_state)
        self.data_dim = X.shape[1]
        self.classes_ = np.unique(y)
        self.y = self._to_tensor(
            self.y_scaler.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        ).reshape(
            -1,
        )
        self.X = self._to_tensor(self.x_scaler.fit_transform(X).astype(np.float32))
        self.fte = (
            self.feature_extractor(data_dim=data_dim)
            if self.feature_extractor
            else None
        )
        if self.problem in ["binomial"]:
            self.likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(self.device)
            self.model = cGPR(
                self.kernel,
                self.mean_module,
                self.X,
                self.y,
                self.likelihood,
                self.random_state,
                feature_extractor=self.fte,
                add_feature=self.add_feature,
                fit_intercept=self.fit_intercept,
            ).to(self.device)
        elif self.problem in ["regression"]:
            if self.alpha is not None:
                self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                    noise=torch.full_like(self.y, self.alpha).type(torch.FloatTensor),
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-4),
                    learn_additional_noise=self.learn_additional_noise,

                ).to(self.device)
            else:
                self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-4),
                    learn_additional_noise=self.learn_additional_noise,
                ).to(self.device)
            self.model = GPR(
                self.kernel,
                self.mean_module,
                self.X,
                self.y,
                self.likelihood,
                self.random_state,
                self.sparse,
                self.n_inducing,
                self.device,
                feature_extractor=self.fte,
                add_feature=self.add_feature,
                fit_intercept=self.fit_intercept,
            ).to(self.device)
        if self.fte:
            self.model.feature_extractor.apply(self.init_weights)
        self.model.mean_module.apply(self.init_weights)
        self.optimizer = self.base_optimizer(params=list(self.model.parameters()))
        self.optimizer.zero_grad()
        if self.problem in ["binomial"]:
            self.mllgp = gpytorch.mlls.VariationalELBO(
                self.likelihood, self.model, self.y.numel()
            )
        elif self.problem in ["regression"]:
            self.mllgp = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood, self.model
            )
        if self.scheduler is not None:
            self.scheduler = self.scheduler(self.optimizer)
        else:
            self.scheduler = NoneStep()

    def _fit(self):
        self.model.train()
        self.likelihood.train()
        self.stopping_criterion = ExpMAStoppingCriterion()
        stop = False
        i = 0

        def closure():
            start = time.time()
            self.optimizer.zero_grad()

            try:  # catch linear algebra errors in gpytorch
                output = self.model(self.X)
                loss = -self.mllgp(output, self.y)
            except RuntimeError as e:
                if "singular" in e.args[0]:
                    return torch.as_tensor(np.nan)
                else:
                    raise e  # pragma: nocover
            reg = 0

            if self.mean_module in [gpytorch.means.LinearMean]:

                if self.l2_reg in [1]:
                    reg += torch.sum(self.model.mean_module.weights ** 2) ** 0.5
                if self.l1_reg in [1]:
                    reg += torch.sum(torch.abs(self.model.mean_module.weights))

            loss += reg
            if self.verbose:
                if i % 25 == 0:
                    print(
                        "Iter %d - Loss: %.3f - Took: %.3f [s]"
                        % (i, loss.item(), time.time() - start)
                    )
            # if loss.requires_grad:
            #   loss.backward()
            return loss

        loss = closure()
        loss.backward()

        if self.partition_kernel:

            # find opt partitions
            N = self.X.size(0)

            # Find the optimum partition/checkpoint size by decreasing in powers of 2
            # Start with no partitioning (size = 0)
            settings = [0] + [
                int(n) for n in np.ceil(N / 2 ** np.arange(1, np.floor(np.log2(N))))
            ]
            for checkpoint_size in settings:
                if self.verbose:
                    print("Kernel partition size: {}".format(checkpoint_size))
                try:
                    # Try a full forward and backward pass with this setting to check memory usage
                    with gpytorch.beta_features.checkpoint_kernel(checkpoint_size):
                        # loss = self.optimizer.step(closure)
                        options = {
                            "closure": closure,
                            "current_loss": loss,
                            "max_ls": 20,
                        }
                        loss, _, lr, _, F_eval, G_eval, _, _ = self.optimizer.step(
                            options
                        )
                    # when successful, break out of for-loop and jump to finally block
                    break
                except RuntimeError as e:
                    if self.verbose:
                        print("RuntimeError: {}".format(e))
                except AttributeError as e:
                    if self.verbose:
                        print("AttributeError: {}".format(e))
                finally:
                    # handle CUDA OOM error
                    gc.collect()
                    torch.cuda.empty_cache()

        if self.partition_kernel:
            while (not stop) & (i < self.max_iter):
                with gpytorch.beta_features.checkpoint_kernel(checkpoint_size):
                    # loss = self.optimizer.step(closure)
                    options = {"closure": closure, "current_loss": loss, "max_ls": 20}
                    loss, _, lr, _, F_eval, G_eval, _, _ = self.optimizer.step(options)
                if self.early_stopping in [1]:
                    stop = self.stopping_criterion.evaluate(fvals=loss.detach().cpu())
                if "ReduceLROnPlateau" in self.scheduler.__class__.__name__:
                    self.scheduler.step(loss.detach().cpu())
                else:
                    self.scheduler.step()
                i += 1

        else:
            while (not stop) & (i < self.max_iter):
                options = {"closure": closure, "current_loss": loss, "max_ls": 10}
                # print(options)
                loss, _, lr, _, F_eval, G_eval, _, _ = self.optimizer.step(options)
                # print(loss)
                if self.early_stopping in [1]:
                    stop = self.stopping_criterion.evaluate(fvals=loss.detach().cpu())

                if "ReduceLROnPlateau" in self.scheduler.__class__.__name__:
                    self.scheduler.step(loss.detach().cpu())
                else:
                    self.scheduler.step()
                i += 1

    def fit(self, X, y):
        if not hasattr(self, "model"):
            """
            Set fit if there is no model object already
            """
            if self.verbose:
                print("Initializing model")
            self._setfit(random_state=self.random_state, X=X, y=y)
        self._fit()
        self.X = None
        self.y = None
        gc.collect()
        torch.cuda.empty_cache()

    def _predict(self, X, sigma=None, n_draws=0):
        torch.cuda.empty_cache()
        set_seed(self.random_state)
        X = self._to_tensor(self.x_scaler.transform(X).astype(np.float32))
        covs_loader = DataLoader(X, batch_size=1024, shuffle=False)
        self.model.eval()
        ar = []

        with torch.no_grad(), gpytorch.settings.fast_pred_samples(), gpytorch.settings.fast_pred_var():
            for data in covs_loader:
                mvnhat = self.likelihood(self.model(data.to(self.device)))
                if n_draws > 0:
                    preds = (
                        mvnhat.sample(sample_shape=torch.Size([int(n_draws)]))
                        .t()
                        .detach()
                        .cpu()
                    )
                else:
                    preds = mvnhat.mean.detach().cpu()
                    if sigma:
                        std = mvnhat.stddev.mul_(sigma).detach().cpu()
                        preds = torch.stack(
                            [preds, preds.sub(std), preds.add(std)], dim=1
                        )
                ar.append(preds)
        # now concat and rescale
        if len(preds.shape) > 1:
            ar = self.y_scaler.inverse_transform(torch.cat(ar, dim=0).cpu().numpy())
        else:
            ar = self.y_scaler.inverse_transform(
                torch.cat(ar, dim=0).cpu().numpy().reshape(-1, 1)
            )

        return ar

    def predict(self, X, sigma=None, n_draws=0):
        hat = self._predict(X, sigma=sigma, n_draws=n_draws)
        return hat

    def init_weights(self, m):
        """
        Usage:
            model = Model()
            model.apply(init_weights)
        """
        if type(m) in [torch.nn.Linear]:
            try:
                self.linear_init(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0)
            except:
                self.linear_init(m.weight.data)
        elif type(m) in [gpytorch.means.LinearMean]:
            try:
                self.linear_init(m.weights.data)
                torch.nn.init.constant_(m.bias.data, 0)
            except:
                self.linear_init(m.weights.data)
        elif type(m) in [
            torch.nn.GRU,
            torch.nn.LSTM,
            torch.nn.RNN,
            torch.nn.GRUCell,
            torch.nn.LSTMCell,
            torch.nn.RNNCell,
        ]:
            for name, param in m.named_parameters():
                if "bias" in name:
                    torch.nn.init.constant_(param.data, 0)
                elif "weight_ih" in name:
                    self.linear_init(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param.data)


class Apollo:
    """
    Fit any sklearn friendly ML model and then fit residuals with a GP.
    """

    def __init__(
        self,
        base_model,
        gpr=GP(mean_module=gpytorch.means.ZeroMean),
    ):
        self.base_model = copy.deepcopy(base_model)
        self.gpr = copy.deepcopy(gpr)

    def fit(self, X, exog, y):
        self.base_model.fit(exog, y)
        residuals = y.reshape(-1, 1) - self.base_model.predict(exog).reshape(-1, 1)
        self.gpr.fit(X, residuals)

    def predict(self, exog, X, sigma=None):
        return self.gpr.predict(X, sigma=sigma) + self.base_model.predict(exog).reshape(
            -1,
        )[:, np.newaxis]
