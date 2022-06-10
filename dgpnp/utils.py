import torch
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kmeans_pytorch import kmeans


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def ls2pr(ls):
    return 1 / (2 * ls**2)


def pr2ls(pr):
    return (1 / (2 * pr)) ** 0.5


def batch_assess(
    model, X, Y, y_scale=None, device="cpu", batch_size=1000, S=100, task="regression"
):
    """
    https://github.com/ICL-SML/Doubly-Stochastic-DGP/blob/master/demos/demo_regression_UCI.ipynb
    """
    if y_scale is None:
        y_scale = 1.0

    n_batches = max(int(X.shape[0] / batch_size), 1)

    if task == "regression":
        all_metrics = {"mnll": [], "mse": []}

    for X_batch, Y_batch in zip(
        np.array_split(X, n_batches), np.array_split(Y, n_batches)
    ):
        _, _, metrics = model.predict(
            X_batch.to(device),
            y=Y_batch.to(device),
            y_scale=y_scale,
            S=S,
        )

        if task == "regression":
            all_metrics["mnll"].append(metrics["mnll"])
            all_metrics["mse"].append(metrics["mse"])

    if task == "regression":
        all_metrics["mse"] = np.mean(np.array(all_metrics["mse"]))
        all_metrics["rmse"] = np.sqrt(all_metrics["mse"])
        all_metrics["mnll"] = np.mean(np.array(all_metrics["mnll"]))

    return all_metrics


def kmeans_initialisations(num_inducing_points, X):
    kmeans = KMeans(n_clusters=num_inducing_points).fit(X)
    return kmeans.cluster_centers_


def gpu_kmeans_initialisations(num_inducing_points, X, device):
    cluster_ids_x, cluster_centers = kmeans(
        X=X, num_clusters=num_inducing_points, distance="euclidean", device=device
    )
    return cluster_centers


def approx_prior_ls(alpha, pg, pu):
    pr = ((alpha + 2 * pg) * pu) / (g_gp.alpha + 2 * (pg + pu))
    return pr2ls(pr)


def approx_prior_norm(alpha, pg, pu):
    return pi / (alpha * (alpha + 2 * (pg + pu))) ** 0.5


def double_integral(xmin, xmax, ymin, ymax, nx, ny, A):
    """
    https://stackoverflow.com/questions/20668689/integrating-2d-samples-on-a-rectangular-grid-using-scipy
    """

    dS = ((xmax - xmin) / (nx - 1)) * ((ymax - ymin) / (ny - 1))

    A_Internal = A[:, 1:-1, 1:-1]

    # sides: up, down, left, right
    (A_u, A_d, A_l, A_r) = (
        A[:, 0, 1:-1],
        A[:, -1, 1:-1],
        A[:, 1:-1, 0],
        A[:, 1:-1, -1],
    )

    # corners
    (A_ul, A_ur, A_dl, A_dr) = (
        A[:, 0, 0],
        A[:, 0, -1],
        A[:, -1, 0],
        A[:, -1, -1],
    )  # dt size vector

    return dS * (
        torch.sum(A_Internal, axis=(1, 2))
        + 0.5
        * (
            torch.sum(A_u, axis=1)
            + torch.sum(A_d, axis=1)
            + torch.sum(A_l, axis=1)
            + torch.sum(A_r, axis=1)
        )
        + 0.25 * (A_ul + A_ur + A_dl + A_dr)
    )


def init_layers(
    layer_class,
    layer_dims,
    X,
    u_inducing_inputs,
    num_outputs_per_layer,
    init_filter_width,
    mc_samples,
    device,
    layer_type="npgp",
    small_internal_cov=False,
    **kwargs,
):
    """Initialisation of the layers within the model."""

    layers = torch.nn.ModuleList([])
    num_layers = len(layer_dims)

    if small_internal_cov:
        init_cov_factors = [1e-5] * (num_layers - 1) + [1.0]
    else:
        init_cov_factors = [1.0] * num_layers

    # Unless all layer input dims are the same, we'll need to project the layer inputs into lower or higher dimensions
    # in order to initialise the mean functions & input process inducing locations in the internal layers
    if (X is None) and (len(set(layer_dims)) != 1):
        raise Exception(
            "If no set of training inputs is supplied for initialising mean functions, all elements in layer_dims must be equal."
        )

    # case no input data and all layers equal
    elif X is None:
        layers = torch.nn.ModuleList(
            [
                layer_class(
                    num_outputs_per_layer[i],
                    u_inducing_inputs,
                    W=None,
                    init_filter_width=init_filter_width[i],
                    mc_samples=mc_samples,
                    device=device,
                    prior_cov_factor_u=init_cov_factors[i],
                    **kwargs,
                )
                for i in range(num_layers)
            ]
        )

    else:
        # If X is not small, initialise using a random subset of training input
        if X.shape[0] > 10000:
            rand_idx = torch.randint(X.shape[0], size=(10000,), device=device)
            X = X[rand_idx, :]

        X_running = X.detach().clone()
        Z_running = u_inducing_inputs.detach().clone()
        for i in range(num_layers):
            d_in = layer_dims[i]
            d_out = num_outputs_per_layer[i]

            # Denote if this is the final layer
            if (i + 1) == num_layers:
                is_final_layer = True
            else:
                is_final_layer = False

            # If layer input and output dims match, no need to compute W
            if d_in == d_out:
                W = None

            # Initialise mean function using PCA projection if we need to step down
            elif d_in >= d_out:
                _, _, V = torch.linalg.svd(X_running, full_matrices=False)
                W = V[:d_out, :].T

            # Initialise using padding if we need to step up
            else:
                W = torch.cat(
                    [
                        torch.eye(d_in, requires_grad=False, device=device),
                        torch.zeros(
                            (d_in, d_out - d_in), requires_grad=False, device=device
                        ),
                    ],
                    1,
                )

            layers.append(
                layer_class(
                    d_out,
                    Z_running,
                    W=W,
                    is_final_layer=is_final_layer,
                    init_filter_width=init_filter_width[i],
                    mc_samples=mc_samples,
                    device=device,
                    prior_cov_factor_u=init_cov_factors[i],
                    **kwargs,
                )
            )

            if d_in != d_out:
                Z_running = Z_running.matmul(W)
                X_running = X_running.matmul(W)

    return layers
