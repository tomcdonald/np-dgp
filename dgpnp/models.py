import torch

torch.set_default_dtype(torch.float64)

import time
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint_sequential
from dgpnp.layers import NPGPLayer, FastNPGPLayer
from dgpnp.utils import (
    init_layers,
    to_numpy,
    batch_assess,
)
from sklearn.metrics import roc_auc_score
import wandb
import torch.nn.functional as F
from math import pi, log, ceil


class NPDeepGP(torch.nn.Module):
    def __init__(
        self,
        u_inducing_inputs,
        num_outputs,
        num_layers,
        X,
        init_filter_width=[0.25, 0.25, 0.25],
        init_noise=0.001,
        mc_samples=10,
        device="cpu",
        small_internal_cov=False,
        fast=True,
        **kwargs,
    ):
        super(NPDeepGP, self).__init__()
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.device = device
        self.mc_samples = mc_samples
        self.N_data = X.shape[0]

        # Set internal layer dimensionalities according to min(d_in, 30)
        d_in = X.shape[1]
        d_internal = min(d_in, 30)
        self.layer_dims = [d_in] + ((num_layers - 1) * [d_internal])
        self.num_outputs_per_layer = self.layer_dims[1:] + [self.num_outputs]

        assert self.layer_dims[0] == u_inducing_inputs.shape[-1]

        if fast:
            layer_class = FastNPGPLayer
        else:
            layer_class = NPGPLayer

        self.layers = init_layers(
            layer_class,
            self.layer_dims,
            X,
            u_inducing_inputs,
            self.num_outputs_per_layer,
            init_filter_width,
            self.mc_samples,
            self.device,
            layer_type="npgp",
            small_internal_cov=small_internal_cov,
            **kwargs,
        )

        self.register_parameter(
            "log_noise",
            torch.nn.Parameter(
                torch.log(torch.tensor([init_noise] * self.num_outputs, device=device))
            ),
        )

    def forward(self, t, checkpoint=False):
        # Ensure input dimensionality == first layer dimensionality
        if len(self.layers) > 1:
            assert t.shape[-1] == self.layer_dims[0]

        if checkpoint:
            layer_out = checkpoint_sequential(self.layers, len(self.layers), t)
        else:
            layer_in = t
            for layer in self.layers:
                layer_in = layer.forward(layer_in)
            layer_out = layer_in

        return layer_out

    def forward_multiple_mc(self, t, S=100):
        """
        Allows for samples to be generated during evaluation
        with mc_multiplier times more MC samples than used during training.
        """
        if S < self.mc_samples:
            mc_multiplier = 1
        else:
            mc_multiplier = int(ceil(S / self.mc_samples))
        all_samps = None
        for i in range(mc_multiplier):
            with torch.no_grad():
                samps = self.forward(t)
                if all_samps is None:
                    all_samps = self.forward(t)
                else:
                    all_samps = torch.cat([all_samps, samps], 0)

        return all_samps[:S]

    def compute_KL(self):
        kl = 0
        for layer in self.layers:
            kl += layer.compute_KL()
        return kl

    def objective(self, x, y, checkpoint=False):
        if checkpoint:
            x = x.requires_grad_(True)  # Required for layer-wise checkpointing to work
        kl = self.compute_KL()
        samps = self.forward(x, checkpoint=checkpoint)

        # If there are missing output observations, compute objective output by output
        if torch.isnan(y).any():
            like = 0.0
            for i in range(self.num_outputs):
                y_d = y[:, i]
                samps_d = samps[:, :, i]

                # Remove missing (NaN) values from target and samples
                is_nan = torch.isnan(y_d)
                y_d = y_d[~is_nan]
                samps_d = samps_d[:, ~is_nan]

                like += torch.sum(
                    torch.mean(
                        (
                            -0.5 * log(2 * pi)
                            - 0.5 * self.log_noise[i]
                            - 0.5
                            * ((y_d - samps_d) ** 2 / torch.exp(self.log_noise[i]))
                        ),
                        0,
                    )
                ) * (self.N_data / samps.shape[1])

        # Otherwise, compute likelihood all at once
        else:
            like = torch.sum(
                torch.mean(
                    (
                        -0.5 * log(2 * pi)
                        - 0.5 * self.log_noise[None, None, :]
                        - 0.5
                        * ((y - samps) ** 2 / torch.exp(self.log_noise[None, None, :]))
                    ),
                    0,
                )
            ) * (self.N_data / samps.shape[1])

        return kl - like

    def get_metrics(self, output, y, y_scale=1.0):
        metrics = {}

        y_pred_mean = torch.mean(output, 0)
        y_pred_std = torch.std(output, 0)

        mnll = -torch.mean(
            torch.mean(
                (
                    -0.5 * log(2 * pi)
                    - 0.5
                    * torch.log(
                        torch.exp(self.log_noise[None, None, :])
                        * y_scale[None, None, :] ** 2
                    )
                    - 0.5
                    * (
                        (y * y_scale - output * y_scale) ** 2
                        / (
                            torch.exp(self.log_noise[None, None, :])
                            * y_scale[None, None, :] ** 2
                        )
                    )
                ),
                0,
            )
        )

        metrics["mnll"] = mnll.item()
        metrics["nmse"] = (
            torch.mean((y_pred_mean - y) ** 2) / torch.mean((torch.mean(y) - y) ** 2)
        ).item()
        metrics["mse"] = (torch.mean(y_scale**2 * (y_pred_mean - y) ** 2)).item()

        return metrics

    def predict(self, x, y=None, y_scale=1.0, S=None):
        with torch.no_grad():

            # Sample self.mc_samples
            if S is None:
                output = self.forward(x)
            # Sample S samples
            else:
                output = self.forward_multiple_mc(x, S=S)

            y_pred_mean = torch.mean(output, 0)
            y_pred_std = torch.std(output, 0)

            if y is not None:
                # If there are missing output observations, compute objective output by output
                if torch.isnan(y).any():
                    metrics_list = []
                    for i in range(self.num_outputs):
                        y_d = y[:, i]
                        output_d = output[:, :, i]

                        # Remove missing (NaN) values from target and samples
                        is_nan = torch.isnan(y_d)
                        y_d = y_d[~is_nan]
                        output_d = output_d[:, ~is_nan]

                        y_pred_d_mean = torch.mean(output_d, 0)
                        y_pred_d_std = torch.std(output_d, 0)

                        metrics_d = self.get_metrics(output_d, y_d, y_scale=y_scale)
                        metrics_list.append(metrics_d)

                        metrics = {}

                        # Average metrics across all outputs
                        for key in metrics_list[0].keys():
                            metrics[key] = sum(
                                [metrics_d[key] for metrics_d in metrics_list]
                            ) / len(metrics_list)

                        return y_pred_mean, y_pred_std, metrics

                # Otherwise, compute metrics all at once
                else:
                    metrics = self.get_metrics(output, y, y_scale=y_scale)
                    return y_pred_mean, y_pred_std, metrics
            else:
                return y_pred_mean, y_pred_std

    def predict_outputs(self, x, y=None, y_scale=1.0, S=None):
        """
        Return metrics output by output (not used in training).
        """
        with torch.no_grad():

            # Sample self.mc_samples
            if S is None:
                output = self.forward(x)
            # Sample S samples
            else:
                output = self.forward_multiple_mc(x, S=S)

            y_pred_mean = torch.mean(output, 0)
            y_pred_std = torch.std(output, 0)

            # If there are missing output observations, compute objective output by output
            if y is not None:
                metrics_list = []
                for i in range(self.num_outputs):
                    y_d = y[:, i]
                    output_d = output[:, :, i]

                    # Remove missing (NaN) values from target and samples
                    if torch.isnan(y_d).any():
                        is_nan = torch.isnan(y_d)
                        y_d = y_d[~is_nan]
                        output_d = output_d[:, ~is_nan]

                    y_pred_d_mean = torch.mean(output_d, 0)
                    y_pred_d_std = torch.std(output_d, 0)

                    metrics_d = self.get_metrics(output_d, y_d, y_scale=y_scale)
                    metrics_list.append(metrics_d)

                metrics = {}

                # Store metrics for all outputs
                for key in metrics_list[0].keys():
                    metrics[key] = [
                        metrics_list[i][key] for i in range(self.num_outputs)
                    ]

                return y_pred_mean, y_pred_std, metrics

            else:
                return y_pred_mean, y_pred_std

    def eval_step(
        self,
        data,
        data_valid,
        y_scale,
        current_iter,
        steps_per_s,
        train_time,
        obj,
        batch_size,
    ):
        with torch.no_grad():
            X_train, y_train = data
            subset_size = min(self.N_data, 5 * batch_size)
            subset_idx = torch.randint(
                y_train.shape[0],
                size=(subset_size,),
                requires_grad=False,
                device=self.device,
            )
            train_metrics = batch_assess(
                self,
                X_train[subset_idx, :],
                y_train[subset_idx, :],
                device=self.device,
                y_scale=y_scale,
                S=5,
            )
            metrics = batch_assess(
                self,
                *data_valid,
                device=self.device,
                y_scale=y_scale,
                S=5,
            )
        print("Iteration %d" % (current_iter))
        print(
            "Steps/s = %.4f, Time = %.4f, Bound = %.4f, Train RMSE = %.4f, Validation RMSE = %.4f, Validation MNLL = %.4f\n"
            % (
                steps_per_s,
                train_time / 1000,
                obj.item(),
                train_metrics["rmse"],
                metrics["rmse"],
                metrics["mnll"],
            )
        )
        wandb.log(
            {
                "iter": current_iter,
                "train rmse": train_metrics["rmse"],
                "val rmse": metrics["rmse"],
                "val mnll": metrics["mnll"],
                "bound": obj.item(),
            }
        )

        return metrics, train_metrics

    def train(
        self,
        data,
        data_valid=None,
        n_iter=100,
        lr=1e-3,
        verbosity=1,
        batch_size=128,
        grad_accum_steps=1,
        chunks=1,
        train_time_limit=None,
        y_scale=None,
        fix_g_pars=False,
        checkpoint=True,
        save_best_model=False,
        model_filepath="model.torch",
        fix_noise_iter=10000,
    ):
        train_time = 0

        # If no validation set specified, just evaluate on the training data
        if data_valid is None:
            data_valid = data

        # Set optimiser and parameters to be optimised
        pars = dict(self.named_parameters())
        for p in list(pars):
            if ("process_kernel.raw_lengthscale") in p:
                pars.pop(p, None)
            if fix_g_pars:
                if ("g_gps" in p) and not ("input" in p):
                    pars.pop(p, None)
            if ("noise") in p:
                pars.pop(p, None)
                fitting_noise = False
        opt = torch.optim.Adam(pars.values(), lr=lr)

        # Initialise dataloader for minibatch training
        train_dataset = torch.utils.data.TensorDataset(*data)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        times_list = []
        start_time = int(round(time.time() * 1000))
        steps_start_time = time.time()
        current_iter = 0
        step_counter = 0
        best_valid_mnll = 1e5
        training = True

        # Perform iterations of minibatch training
        while training:

            if (not fitting_noise) and (current_iter > fix_noise_iter):
                opt.add_param_group({"params": self.log_noise})
                print("\nNow fitting noise...\n")
                fitting_noise = True

            # Use each batch to train model
            for X_minibatch, y_minibatch in train_dataloader:

                if chunks > 1:
                    X_chunks = torch.tensor_split(X_minibatch.to(self.device), chunks)
                    y_chunks = torch.tensor_split(y_minibatch.to(self.device), chunks)

                for i in range(grad_accum_steps):
                    if chunks > 1:
                        obj = self.objective(
                            X_chunks[0],
                            y_chunks[0],
                            checkpoint=checkpoint,
                        )
                        for i in range(1, chunks):
                            obj += self.objective(
                                X_chunks[i],
                                y_chunks[i],
                                checkpoint=checkpoint,
                            )
                        obj /= grad_accum_steps
                    else:
                        obj = (
                            self.objective(
                                X_minibatch.to(self.device),
                                y_minibatch.to(self.device),
                                checkpoint=checkpoint,
                            )
                            / grad_accum_steps
                        )
                    obj.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                step_counter += 1
                current_iter += 1

                # Stop training after time limit elapsed
                if train_time_limit is not None:
                    if train_time > 1000 * 60 * train_time_limit:
                        print("\nTraining Complete.\n")
                        training = False
                        break

                # Stop training if specified number of iterations has been completed
                if current_iter >= n_iter:
                    print("\nTraining Complete.\n")
                    training = False
                    break

                # Display validation metrics at specified intervals
                if verbosity == 0:
                    if current_iter % 100 == 0:
                        print("%d training iterations completed." % current_iter)
                elif current_iter % verbosity == 0:
                    steps_per_s = step_counter / (time.time() - steps_start_time)

                    metrics, _ = self.eval_step(
                        data,
                        data_valid,
                        y_scale,
                        current_iter,
                        steps_per_s,
                        train_time,
                        obj,
                        batch_size,
                    )
                    times_list.append(train_time / 1000)
                    step_counter = 0
                    steps_start_time = time.time()
                    train_time = int(round(time.time() * 1000)) - start_time

                    # If saving best model & validation MNLL is best recorded so far, save model state
                    if save_best_model and (metrics["mnll"] < best_valid_mnll):
                        best_valid_mnll = metrics["mnll"]
                        torch.save(self.state_dict(), model_filepath)

        # If not saving best model during training, save the final model once training complete
        if not save_best_model:
            torch.save(self.state_dict(), model_filepath)

    def assess_layer_power(self):
        "Returns tensor of internal layer 'power' values, relative to output layer."
        with torch.no_grad():
            layer_powers = []
            final_dummy_input = torch.randn(
                100, self.layer_dims[-1], requires_grad=False, device=self.device
            )
            final_dummy_output = self.layers[-1].forward(final_dummy_input)
            final_layer_power = torch.sum(torch.square(final_dummy_output)).item()
            for i in range(self.num_layers - 1):
                dummy_input = torch.randn(
                    100, self.layer_dims[i], requires_grad=False, device=self.device
                )
                dummy_output = self.layers[i].forward(dummy_input)
                layer_powers.append(
                    torch.sum(torch.square(dummy_output)).item()
                    / (dummy_output.shape[-1] * final_layer_power)
                )
        return layer_powers

    def plot_features(self, save=None):

        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                name = save[:-4] + f"_layer_{i}" + save[-4:]
                layer.plot_features(save=name)
                name = save[:-4] + f"_layer_amps{i}" + save[-4:]
                layer.plot_amps(save=name)

    def plot_predictions(self, X, y, save=None, S=20):

        with torch.no_grad():

            m, std = self.predict(X, S=S)

            if X.shape < y.shape:
                y_vert = True
                fig, axs = plt.subplots(
                    y.shape[1], X.shape[1], figsize=(10, y.shape[1] * 3), squeeze=False
                )

            else:
                y_vert = False
                fig, axs = plt.subplots(
                    X.shape[1], y.shape[1], figsize=(10, 3 * X.shape[1]), squeeze=False
                )

            for j in range(y.shape[1]):
                for i in range(X.shape[1]):
                    sort_idx = torch.argsort(X[:, i])

                    mi = m[sort_idx, j].detach().cpu().numpy().flatten()
                    si = std[sort_idx, j].detach().cpu().numpy().flatten()
                    Xi = X[sort_idx, i].detach().cpu().numpy()
                    try:
                        Zs = (
                            self.layers[0]
                            .u_gp.inducing_inputs[:, i]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    except AttributeError:
                        Zs = (
                            self.layers[0]
                            .gps[0]
                            .inducing_inputs[:, i]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                    if y_vert:
                        ax = axs[j, i]
                    else:
                        ax = axs[i, j]

                    ax.plot(Xi, mi)
                    ax.fill_between(Xi, mi - si, mi + si, alpha=0.25)

                    if y is not None:
                        yi = y[sort_idx, j].detach().cpu().numpy().flatten()
                        ax.scatter(Xi, yi, alpha=0.25)
                        ax.vlines(
                            Zs,
                            0,
                            1,
                            alpha=0.25,
                            color="red",
                            transform=ax.get_xaxis_transform(),
                        )

            if save is not None:
                plt.savefig(save)
            else:
                plt.show()

    def plot_output_slices(self, X, save=None):
        with torch.no_grad():
            dims = X.shape[1]
            xsi = torch.linspace(
                -torch.max(X),
                torch.max(X),
                500,
                requires_grad=False,
                device=self.device,
            )

            fig, axs = plt.subplots(dims, 1, figsize=(6, 2 * dims), squeeze=False)
            for i in range(dims):
                xl = [torch.zeros((500, 1), device=self.device)] * dims
                xl[i] = xsi[:, None]
                x_slice = torch.hstack(xl)
                out = self.forward(x_slice)
                axs[i][0].plot(to_numpy(xsi), to_numpy(out)[:, :, 0].T)

            if save is not None:
                plt.savefig(save)
            else:
                plt.show()
