"""Utility functions for Gaussian Processes."""

import casadi as ca
import gpytorch
import numpy as np
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from numpy.typing import NDArray

torch.manual_seed(0)


# TODO: Currently not used, marked for removal.
def covSEard(x, z, ell, sf2):
    """GP squared exponential kernel.

    This function is based on the 2018 GP-MPC library by Helge-André Langåker

    Args:
        x (np.array or casadi.MX/SX): First vector.
        z (np.array or casadi.MX/SX): Second vector.
        ell (np.array or casadi.MX/SX): Length scales.
        sf2 (float or casadi.MX/SX): output scale parameter.

    Returns:
        SE kernel (casadi.MX/SX): SE kernel.
    """
    dist = ca.sum1((x - z) ** 2 / ell**2)
    return sf2 * ca.SX.exp(-0.5 * dist)


def covSE_single(x, z, ell, sf2):
    dist = ca.sum1((x - z) ** 2 / ell**2)
    return sf2 * ca.exp(-0.5 * dist)


# TODO: Currently not used, marked for removal.
def covMatern52ard(x, z, ell, sf2):
    """Matern kernel that takes nu equal to 5/2.

    Args:
        x (np.array or casadi.MX/SX): First vector.
        z (np.array or casadi.MX/SX): Second vector.
        ell (np.array or casadi.MX/SX): Length scales.
        sf2 (float or casadi.MX/SX): output scale parameter.

    Returns:
        Matern52 kernel (casadi.MX/SX): Matern52 kernel.

    """
    dist = ca.sum1((x - z) ** 2 / ell**2)
    r_over_l = ca.sqrt(dist)
    return sf2 * (1 + ca.sqrt(5) * r_over_l + 5 / 3 * r_over_l**2) * ca.exp(-ca.sqrt(5) * r_over_l)


class GPRegressionModel(gpytorch.models.ExactGP):
    """Single dimensional output Gaussian Process model with zero mean function and RBF kernel."""

    def __init__(
        self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: GaussianLikelihood
    ):
        """Initialize a single dimensional Gaussian Process model with zero mean function.

        Args:
            train_x: input training data (input_dim X N samples).
            train_y: output training data (output dim x N samples).
            likelihood: likelihood function.
        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean().to(train_x.device)
        self.covar_module = ScaleKernel(RBFKernel()).to(train_x.device)
        # Materialize the covariance matrix
        self.K = self.covar_module(train_x).add_diag(self.likelihood.noise).to_dense()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_gp(
    model: GPRegressionModel,
    likelihood: GaussianLikelihood,
    iterations: int = 500,
    learning_rate: float = 0.01,
    device: str = "cpu",
):
    assert model.train_inputs is not None, "model train inputs must be set at initialization"
    assert model.train_targets is not None, "model train targets must be set at initialization"
    train_x = model.train_inputs[0].to(device)
    train_y = model.train_targets[0].to(device)
    model = model.to(device)
    model.train()
    likelihood = likelihood.to(device)
    likelihood.train()

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    last_loss = torch.tensor(torch.inf)
    for _ in range(iterations):
        optim.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optim.step()
        if torch.abs(last_loss - loss) < 1e-3:  # Early stopping if converged
            break
        last_loss = loss


def gpytorch_predict2casadi(
    model: GPRegressionModel, train_x: NDArray, train_y: NDArray
) -> ca.Function:
    """Convert the prediction function of a gpytorch model to casadi model."""
    assert isinstance(model, GPRegressionModel), "Can only convert GPRegressionModels to casadi"
    assert train_x.ndim == 2, "train_x must be a 2D array"
    assert train_y.ndim == 2, "train_y must be a 2D array"
    lengthscale = model.covar_module.base_kernel.lengthscale.numpy(force=True)
    output_scale = model.covar_module.outputscale.numpy(force=True)
    Nx = train_x.shape[1]
    z = ca.SX.sym("z", Nx)
    K_z_ztrain = ca.Function(
        "k_z_ztrain",
        [z],
        [covSE_single(z, train_x.T, lengthscale.T, output_scale)],
        ["z"],
        ["K"],
    )
    K = model.covar_module(train_x).add_diag(model.likelihood.noise)
    print(K, type(K))
    K_noise_inv = K.inv_matmul(torch.eye(n_samples).double())
    predict = ca.Function(
        "pred",
        [z],
        [K_z_ztrain(z=z)["K"] @ self.model.K_plus_noise_inv.numpy(force=True) @ train_y],
        ["z"],
        ["mean"],
    )
    return predict


class GaussianProcess:
    """Gaussian Process decorator for gpytorch."""

    def __init__(self, model_type, likelihood, kernel="RBF"):
        """Initialize Gaussian Process.

        Args:
            model_type (gpytorch model class): Model class for the GP (ZeroMeanIndependentMultitaskGPModel).
            likelihood (gpytorch.likelihood): likelihood function.
        """
        self.model_type = model_type
        self.likelihood = likelihood
        self.optimizer = None
        self.model = None
        self.kernel = kernel
        self.input_dimension = None
        self.output_dimension = None
        self.n_training_samples = None

    def _init_model(self, train_inputs, train_targets):
        """Init GP model from train inputs and train_targets."""
        target_dimension = train_targets.shape[1] if train_targets.ndim > 1 else 1
        input_dimension = train_inputs.shape[1] if train_inputs.ndim > 1 else 1

        self.model = self.model_type(
            train_inputs, train_targets, self.likelihood, kernel=self.kernel
        )
        # Extract dimensions for external use.
        self.input_dimension = input_dimension
        self.output_dimension = target_dimension
        self.n_training_samples = train_inputs.shape[0]

    def _compute_GP_covariances(self, train_x):
        """Compute K(X,X) + sigma*I and its inverse."""
        # Pre-compute inverse covariance plus noise to speed-up computation.
        K_lazy = self.model.covar_module(train_x.double())
        K_lazy_plus_noise = K_lazy.add_diag(self.model.likelihood.noise)
        n_samples = train_x.shape[0]
        self.model.K_plus_noise = K_lazy_plus_noise.matmul(torch.eye(n_samples).double())
        self.model.K_plus_noise_inv = K_lazy_plus_noise.inv_matmul(torch.eye(n_samples).double())

    def train(
        self, train_inputs, train_targets, n_train=500, learning_rate=0.01, device: str = "cpu"
    ):
        """Train the GP using Train_x and Train_y.

        Args:
            train_x: Torch tensor (N samples [rows] by input dim [cols])
            train_y: Torch tensor (N samples [rows] by target dim [cols])
        """
        self._init_model(train_inputs, train_targets)
        train_x = train_inputs.to(device)
        train_y = train_targets.to(device)
        self.model = self.model.to(device)
        self.likelihood = self.likelihood.to(device)

        if self.input_dimension == 1:
            train_x = train_x.reshape(-1)

        self.model.double()
        self.likelihood.double()
        self.model.train()
        self.likelihood.train()

        optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        last_loss = torch.tensor(torch.inf)
        for i in range(n_train):
            optim.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optim.step()
            if torch.abs(last_loss - loss) < 1e-3:  # Early stopping if converged
                break
            last_loss = loss

        self.model = self.model.cpu()
        self.likelihood = self.likelihood.cpu()
        train_x = train_x.cpu()
        train_y = train_y.cpu()
        self._compute_GP_covariances(train_x)
        self.casadi_predict = self.make_casadi_prediction_func(train_x, train_y)

    def predict(self, x, requires_grad=False, return_pred=True):
        """
        Args:
            x : torch.Tensor (N_samples x input DIM).

        Returns:
            Predictions
                mean : torch.tensor (nx X N_samples).
                lower : torch.tensor (nx X N_samples).
                upper : torch.tensor (nx X N_samples).
        """
        self.model.eval()
        self.likelihood.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).double()
        if requires_grad:
            predictions = self.likelihood(self.model(x))
            mean = predictions.mean
            cov = predictions.covariance_matrix
        else:
            with torch.no_grad(), gpytorch.settings.fast_pred_var(
                state=True
            ), gpytorch.settings.fast_pred_samples(state=True):
                predictions = self.likelihood(self.model(x))
                mean = predictions.mean
                cov = predictions.covariance_matrix
        if return_pred:
            return mean, cov, predictions
        else:
            return mean, cov

    def make_casadi_prediction_func(self, train_inputs, train_targets):
        """Assumes train_inputs and train_targets are already masked."""
        train_inputs = train_inputs.numpy()
        train_targets = train_targets.numpy()
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach().numpy()
        output_scale = self.model.covar_module.outputscale.detach().numpy()
        Nx = self.input_dimension
        if train_targets.ndim == 1:
            train_targets = train_targets.reshape(-1, 1)
        if train_inputs.ndim == 1:
            train_inputs = train_inputs.reshape(-1, 1)
        z = ca.SX.sym("z", Nx)
        if self.kernel == "RBF":
            K_z_ztrain = ca.Function(
                "k_z_ztrain",
                [z],
                [covSEard(z, train_inputs.T, lengthscale.T, output_scale)],
                ["z"],
                ["K"],
            )
        elif self.kernel == "RBF_single":
            K_z_ztrain = ca.Function(
                "k_z_ztrain",
                [z],
                [covSE_single(z, train_inputs.T, lengthscale.T, output_scale)],
                ["z"],
                ["K"],
            )
        elif self.kernel == "Matern":
            K_z_ztrain = ca.Function(
                "k_z_ztrain",
                [z],
                [covMatern52ard(z, train_inputs.T, lengthscale.T, output_scale)],
                ["z"],
                ["K"],
            )
        predict = ca.Function(
            "pred",
            [z],
            [K_z_ztrain(z=z)["K"] @ self.model.K_plus_noise_inv.detach().numpy() @ train_targets],
            ["z"],
            ["mean"],
        )
        return predict
