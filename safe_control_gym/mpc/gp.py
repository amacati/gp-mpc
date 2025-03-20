"""Utility functions for Gaussian Processes."""

import casadi as cs
import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean


def covSE_single(x, z, ell, sf2):
    dist = cs.sum1((x - z) ** 2 / ell**2)
    return sf2 * cs.exp(-0.5 * dist)


def covSE_vectorized(x, Z, ell, sf2):
    """Vectorized kernel version of covSE_single."""
    x_reshaped = cs.repmat(x, 1, Z.shape[0])  # Reshape x to match Z's dimensions for broadcasting
    dist = cs.sum1((x_reshaped - Z.T) ** 2 / ell**2)
    return sf2 * cs.exp(-0.5 * dist)


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
        self.K, self.K_inv = None, None  # Only computed once the GP is trained

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def compute_covariances(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute K(X,X) + sigma*I and its inverse."""
        K = self.covar_module(self.train_inputs[0]).add_diag(self.likelihood.noise).to_dense()
        return K, torch.linalg.inv(K)


def fit_gp(gp: GaussianProcess, n_train=500, lr=0.01, device: str = "cpu"):
    """Fit a GP to its training data."""
    assert isinstance(gp, GaussianProcess), f"gp must be a GaussianProcess, got {type(gp)}"
    train_x = gp.train_inputs[0].to(device)
    train_y = gp.train_targets.to(device)

    gp.train().to(device)
    optim = torch.optim.Adam(gp.parameters(), lr=lr)
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

    gp.K, gp.K_inv = gp.compute_covariances()


def gpytorch_predict2casadi(gp: GaussianProcess) -> cs.Function:
    """Convert the prediction function of a gpytorch model to casadi model."""
    assert isinstance(gp, GaussianProcess), f"Expected a GaussianProcess, got {type(gp)}"
    train_inputs = gp.train_inputs[0].numpy(force=True)
    train_targets = gp.train_targets.numpy(force=True)
    assert train_inputs.ndim == 2, "train_inputs must be a 2D array"
    lengthscale = gp.covar_module.base_kernel.lengthscale.to_dense().numpy(force=True)
    output_scale = gp.covar_module.outputscale.to_dense().numpy(force=True)

    z = cs.SX.sym("z", train_inputs.shape[1])
    kernel_fn = covSE_single(z, train_inputs.T, lengthscale.T, output_scale)
    K_xz = cs.Function("K_xz", [z], [kernel_fn], ["z"], ["K"])
    K_xx_inv = gp.K_inv.numpy(force=True)
    return cs.Function("pred", [z], [K_xz(z=z)["K"] @ K_xx_inv @ train_targets], ["z"], ["mean"])
