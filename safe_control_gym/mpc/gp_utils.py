"""Utility functions for Gaussian Processes."""

import casadi as ca
import gpytorch
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from numpy.typing import NDArray


def covSE_single(x, z, ell, sf2):
    dist = ca.sum1((x - z) ** 2 / ell**2)
    return sf2 * ca.exp(-0.5 * dist)


class GaussianProcess(gpytorch.models.ExactGP):
    """Gaussian Process decorator for gpytorch."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        """Initialize Gaussian Process."""
        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor"
        assert isinstance(y, torch.Tensor), "y must be a torch.Tensor"
        likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
        super().__init__(x, y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())
        # Save dimensions for later use
        self.n_ind_points = self.train_inputs[0].shape[0]
        self.input_dimension = self.train_inputs[0].shape[1]
        self.K_plus_noise = None  # Only computed once the GP is trained
        self.K_plus_noise_inv = None

    def compute_GP_covariances(self):
        """Compute K(X,X) + sigma*I and its inverse."""
        # Pre-compute inverse covariance plus noise to speed-up computation.
        K = self.covar_module(self.train_inputs[0]).add_diag(self.likelihood.noise)
        self.K_plus_noise = K.to_dense()
        self.K_plus_noise_inv = torch.linalg.inv(self.K_plus_noise)

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def predict(self, x):
        """
        Args:
            x : torch.Tensor (N_samples x input DIM).

        Returns:
            Predictions
                mean : torch.tensor (nx X N_samples).
                lower : torch.tensor (nx X N_samples).
                upper : torch.tensor (nx X N_samples).
        """
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(self.train_inputs[0].device)
        with (
            torch.no_grad(),
            gpytorch.settings.fast_pred_var(state=True),
            gpytorch.settings.fast_pred_samples(state=True),
        ):
            predictions = self.likelihood(self(x))
            mean = predictions.mean
            cov = predictions.covariance_matrix
        return mean.numpy(force=True), cov.numpy(force=True)

    def make_casadi_prediction_func(self, train_inputs, train_targets):
        """Assumes train_inputs and train_targets are already masked."""
        train_inputs = train_inputs.numpy()
        train_targets = train_targets.numpy()
        lengthscale = self.covar_module.base_kernel.lengthscale.detach().numpy()
        output_scale = self.covar_module.outputscale.detach().numpy()
        Nx = self.input_dimension
        if train_targets.ndim == 1:
            train_targets = train_targets.reshape(-1, 1)
        if train_inputs.ndim == 1:
            train_inputs = train_inputs.reshape(-1, 1)
        z = ca.SX.sym("z", Nx)

        K_z_ztrain = ca.Function(
            "k_z_ztrain",
            [z],
            [covSE_single(z, train_inputs.T, lengthscale.T, output_scale)],
            ["z"],
            ["K"],
        )
        predict = ca.Function(
            "pred",
            [z],
            [K_z_ztrain(z=z)["K"] @ self.K_plus_noise_inv.detach().numpy() @ train_targets],
            ["z"],
            ["mean"],
        )
        return predict


def fit_gp(gp: GaussianProcess, n_train=500, learning_rate=0.01, device: str = "cpu"):
    """Fit a GP to its training data."""
    assert isinstance(gp, GaussianProcess), f"gp must be a GaussianProcess, got {type(gp)}"
    train_x = gp.train_inputs[0].to(device)
    train_y = gp.train_targets.to(device)

    gp.train().to(device)
    optim = torch.optim.Adam(gp.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    last_loss = torch.tensor(torch.inf)
    for _ in range(n_train):
        optim.zero_grad()
        output = gp(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optim.step()
        if torch.abs(last_loss - loss) < 1e-3:  # Early stopping if converged
            break
        last_loss = loss

    gp.compute_GP_covariances()


def gpytorch_predict2casadi(gp: GaussianProcess, train_x: NDArray, train_y: NDArray) -> ca.Function:
    """Convert the prediction function of a gpytorch model to casadi model."""
    assert isinstance(gp, GaussianProcess), f"Expected a GaussianProcess, got {type(gp)}"
    assert train_x.ndim == 2, "train_x must be a 2D array"
    assert train_y.ndim == 2, "train_y must be a 2D array"
    lengthscale = gp.covar_module.base_kernel.lengthscale.numpy(force=True)
    output_scale = gp.covar_module.outputscale.numpy(force=True)
    Nx = train_x.shape[1]
    z = ca.SX.sym("z", Nx)
    K_z_ztrain = ca.Function(
        "k_z_ztrain",
        [z],
        [covSE_single(z, train_x.T, lengthscale.T, output_scale)],
        ["z"],
        ["K"],
    )
    K = gp.covar_module(train_x).add_diag(gp.likelihood.noise).numpy(force=True)
    predict = ca.Function(
        "pred",
        [z],
        [K_z_ztrain(z=z)["K"] @ np.linalg.solve(K @ train_y)],
        ["z"],
        ["mean"],
    )
    return predict
