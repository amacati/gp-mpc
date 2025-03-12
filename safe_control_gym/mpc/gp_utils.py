"""Utility functions for Gaussian Processes."""
from pathlib import Path

import casadi as ca
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans

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


class ZeroMeanIndependentGPModel(gpytorch.models.ExactGP):
    """Single dimensional output Gaussian Process model with zero mean function.

    Or constant mean and radial basis function kernel (SE).
    """

    def __init__(self, train_x, train_y, likelihood, kernel="RBF"):
        """Initialize a single dimensional Gaussian Process model with zero mean function.

        Args:
            train_x (torch.Tensor): input training data (input_dim X N samples).
            train_y (torch.Tensor): output training data (output dim x N samples).
            likelihood (gpytorch.likelihood): Likelihood function (gpytorch.likelihoods.GaussianLikelihood).
        """
        super().__init__(train_x, train_y, likelihood)
        # For Zero mean function.
        self.mean_module = gpytorch.means.ZeroMean()
        # For constant mean function.
        if kernel == "RBF":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]),
                ard_num_dims=train_x.shape[1],
            )
        elif kernel == "Matern":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[1]),
                ard_num_dims=train_x.shape[1],
            )
        elif kernel == "RBF_single":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(),
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcess:
    """Gaussian Process decorator for gpytorch."""

    def __init__(
        self,
        model_type,
        likelihood,
        input_mask=None,
        target_mask=None,
        kernel="RBF",
        noise_prior=None,
    ):
        """Initialize Gaussian Process.

        Args:
            model_type (gpytorch model class): Model class for the GP (ZeroMeanIndependentMultitaskGPModel).
            likelihood (gpytorch.likelihood): likelihood function.
        """
        self.model_type = model_type
        self.likelihood = likelihood
        self.optimizer = None
        self.model = None
        self.input_mask = input_mask
        self.target_mask = target_mask
        self.kernel = kernel
        self.noise_prior = noise_prior

    def _init_model(self, train_inputs, train_targets):
        """Init GP model from train inputs and train_targets."""
        target_dimension = train_targets.shape[1] if train_targets.ndim > 1 else 1
        input_dimension = train_inputs.shape[1] if train_inputs.ndim > 1 else 1

        if self.model is None:
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

    def init_with_hyperparam(self, train_inputs, train_targets, path_to_statedict):
        """Load hyperparameters from a state_dict."""
        if self.input_mask is not None:
            train_inputs = train_inputs[:, self.input_mask]
        if self.target_mask is not None:
            train_targets = train_targets[:, self.target_mask]
        device = torch.device("cpu")
        state_dict = torch.load(path_to_statedict, map_location=device)
        self._init_model(train_inputs, train_targets)
        self.model.load_state_dict(state_dict)
        self.model.double()  # needed otherwise loads state_dict as float32
        self._compute_GP_covariances(train_inputs)
        self.casadi_predict = self.make_casadi_prediction_func(train_inputs, train_targets)

    def train(
        self,
        train_input_data,
        train_target_data,
        test_input_data,
        test_target_data,
        n_train=500,
        learning_rate=0.01,
        gpu=False,
        fname: Path = Path("best_model.pth"),
    ):
        """Train the GP using Train_x and Train_y.

        Args:
            train_x: Torch tensor (N samples [rows] by input dim [cols])
            train_y: Torch tensor (N samples [rows] by target dim [cols])
        """
        train_x_raw = train_input_data
        train_y_raw = train_target_data
        test_x_raw = test_input_data
        test_y_raw = test_target_data
        if self.input_mask is not None:
            train_x_raw = train_x_raw[:, self.input_mask]
            test_x_raw = test_x_raw[:, self.input_mask]
        if self.target_mask is not None:
            train_y_raw = train_y_raw[:, self.target_mask]
            test_y_raw = test_y_raw[:, self.target_mask]
        self._init_model(train_x_raw, train_y_raw)
        train_x = train_x_raw
        train_y = train_y_raw
        test_x = test_x_raw
        test_y = test_y_raw
        if gpu:
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            test_x = test_x.cuda()
            test_y = test_y.cuda()
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

        if self.input_dimension == 1:
            test_x = test_x.reshape(-1)
            train_x = train_x.reshape(-1)

        self.model.double()
        self.likelihood.double()
        self.model.train()
        self.likelihood.train()

        max_trial = 1
        opti_result = []
        loss_result = []
        for trial_idx in range(max_trial):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
            last_loss = 99999999
            best_loss = 99999999
            loss = torch.tensor(0)
            i = 0
            while i < n_train and torch.abs(loss - last_loss) > 1e-2:
                with torch.no_grad():
                    self.model.eval()
                    self.likelihood.eval()
                    test_output = self.model(test_x)
                    test_loss = -mll(test_output, test_y)
                self.model.train()
                self.likelihood.train()
                self.optimizer.zero_grad()
                output = self.model(train_x)
                loss = -mll(output, train_y)
                loss.backward()

                self.optimizer.step()
                if test_loss < best_loss:
                    best_loss = test_loss
                    state_dict = self.model.state_dict()
                    torch.save(state_dict, fname)

                i += 1
            opti_result.append(state_dict)
            loss_result.append(best_loss.cpu().detach().numpy())
        # find and save the best model among the trials.
        best_idx = np.argmin(loss_result)
        torch.save(opti_result[best_idx], fname)
        self.model = self.model.cpu()
        self.likelihood = self.likelihood.cpu()
        train_x = train_x.cpu()
        train_y = train_y.cpu()
        self.model.load_state_dict(torch.load(fname))
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
        if self.input_mask is not None:
            x = x[:, self.input_mask]
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

    def prediction_jacobian(self, query):
        mean_der, _ = torch.autograd.functional.jacobian(
            lambda x: self.predict(x, requires_grad=True, return_pred=False), query.double()
        )
        return mean_der.detach().squeeze()

    def make_casadi_prediction_func(self, train_inputs, train_targets):
        """Assumes train_inputs and train_targets are already masked."""
        train_inputs = train_inputs.numpy()
        train_targets = train_targets.numpy()
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach().numpy()
        output_scale = self.model.covar_module.outputscale.detach().numpy()
        # Nx = len(self.input_mask)
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

    def plot_trained_gp(
        self, inputs, targets, output_label, output_dir=None, fig_count=0, title=None, **kwargs
    ):
        """Plot the trained GP given the input and target data.

        Args:
            inputs (torch.Tensor): Input data (N_samples x input_dim).
            targets (torch.Tensor): Target data (N_samples x 1).
            output_label (str): Label for the output. Usually the index of the output.
            output_dir (str): Directory to save the figure.
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).double()
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).double()

        num_data = inputs.shape[0]
        residual_func = kwargs.get("residual_func", None)
        residual = np.zeros((num_data, targets.shape[1]))
        if residual_func is not None:
            for i in range(num_data):
                residual[i, :] = residual_func(inputs[i, :].numpy())[output_label]

        if self.target_mask is not None:
            targets = targets[:, self.target_mask]
        means, _, preds = self.predict(inputs)
        t = np.arange(inputs.shape[0])
        lower, upper = preds.confidence_region()

        fig_count += 1
        plt.figure(fig_count, figsize=(5, 2))
        # compute the percentage of test points within 2 std
        num_within_2std = torch.sum((targets[:, i] > lower) & (targets[:, i] < upper)).numpy()
        percentage_within_2std = num_within_2std / len(targets[:, i]) * 100
        plt.fill_between(
            t, lower.detach().numpy(), upper.detach().numpy(), alpha=0.5, label="2-$\sigma$"
        )
        plt.plot(t, means, "b", label="GP mean")
        plt.scatter(t, targets, color="r", label="Target")
        plt.plot(
            t,
            residual,
            "g",
            label="Residual",
        )
        plt.legend(ncol=2)
        plt_title = f"GP validation {output_label}, {percentage_within_2std:.2f}% within 2-$\sigma$"
        if title is not None:
            plt_title += f" {title}"
        plt.title(plt_title)

        if output_dir is not None:
            plt_name = (
                f"gp_validation_{output_label}.png"
                if title is None
                else f"gp_validation_{output_label}_{title}.png"
            )
            plt.savefig(f"{output_dir}/{plt_name}")
            print(f"Figure saved at {output_dir}/{plt_name}")
        # clean up the plot
        plt.close(fig_count)

        return fig_count


def kmeans_centroids(n_cent, data, rand_state=0):
    """kmeans clustering. Useful for finding reasonable inducing points.

    Args:
        n_cent (int): Number of centriods.
        data (np.array): Data to find the centroids of n_samples X n_features.

    Return:
        centriods (np.array): Array of centriods (n_cent X n_features).
    """
    kmeans = KMeans(n_clusters=n_cent, random_state=rand_state).fit(data)
    return kmeans.cluster_centers_
