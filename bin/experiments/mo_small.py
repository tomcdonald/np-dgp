import torch

torch.set_default_dtype(torch.float64)
from npdgp.data import Datasets
from npdgp.models import NPDeepGP
from npdgp.utils import (
    batch_assess,
    to_numpy,
    kmeans_initialisations,
)
import numpy as np
import argparse
import os
import pathlib
from datetime import datetime
import wandb


from npdgp.baseline_models import fit_gpytorch_mogp
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


def run(args):
    # Seeds & GPU settings
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("SEED =", SEED)

    str_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    save_dir = os.path.join(args.output_dir, "polymer", str_time)

    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = (
            True  # Let CUDA optimise strategy, as our inputs are not of variable size
        )
        torch.backends.cudnn.enabled = True
    else:
        device = "cpu"

    print("Device:", device, "\n")
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    ds = Datasets(data_path="data/")
    data = ds.all_datasets["polymer"].get_data()
    X_train, y_train, X_test, y_test, y_scale = [
        torch.tensor(data[_], dtype=torch.float64, device=device)
        for _ in ["X", "Y", "Xs", "Ys", "Y_std"]
    ]

    main_kwargs = {
        "num_layers": args.num_layers,
        "num_basis_functions": 16,
        "mc_samples": 2,
        "init_filter_width": [2.0] * (args.num_layers - 1) + [2.0],
        "num_filter_points": 15,
        "init_noise": 1e-2,
        "init_amp": 1.0,
        "small_internal_cov": False,
        "prior_cov_factor_g": 0.8,
        "init_transform_lengthscale": 0.5,
        "jitter": 1e-7,
        "fast": args.fast,
        "beta": 0.8,
        "prior_mean_factor_g": 0.5,
        "device": device,
    }

    train_kwargs = {
        "lr": 1e-3,
        "n_iter": args.n_iter,
        "verbosity": args.verbosity,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "chunks": args.chunks,
        "train_time_limit": args.time,
        "fix_g_pars": False,
        "checkpoint": True,
        "fix_noise_iter": 1,
        "model_filepath": os.path.join(save_dir, "model.torch"),
    }

    num_u_ips = 20
    u_ip_inits = kmeans_initialisations(num_u_ips, to_numpy(X_train))
    u_ip_inits = torch.tensor(u_ip_inits, device=device)

    if args.dry_run:
        wandb_mode = "disabled"
    else:
        wandb_mode = "online"

    config = {**train_kwargs, **main_kwargs, "seed": SEED}

    # Fit GPyTorch MOGP
    baseline_preds, baseline_mnll = fit_gpytorch_mogp(
        X_train,
        y_train,
        X_test,
        y_test,
        u_ip_inits,
        n_iter=1000,
        num_tasks=y_train.shape[1],
        num_latents=y_train.shape[1],
        device=device,
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            to_numpy(y_scale[None, :] * y_test),
            to_numpy(y_scale[None, :] * baseline_preds.mean),
        )
    )
    baseline_mape = mean_absolute_percentage_error(
        to_numpy(y_test), to_numpy(baseline_preds.mean)
    )
    print(f"{baseline_rmse=}")
    print(f"{baseline_mape=}")
    print(f"{baseline_mnll=}")

    wandb.init(
        project="npdgp-mo",
        entity="npdgp",
        config=config,
        mode=wandb_mode,
    )

    npdgp = NPDeepGP(
        u_inducing_inputs=u_ip_inits,
        num_outputs=y_train.shape[1],
        X=X_train,
        **main_kwargs,
    ).to(device)

    wandb.watch(npdgp)

    npdgp.plot_output_slices(X_test, save=os.path.join(save_dir, "prior_slice.png"))
    npdgp.plot_features(save=os.path.join(save_dir, "pre_features.png"))
    npdgp.plot_predictions(
        X_test,
        y_test,
        save=os.path.join(save_dir, "prior_preds.png"),
    )

    npdgp.train(
        (X_train, y_train),
        (X_test, y_test),
        y_scale=y_scale,
        **train_kwargs,
    )

    pred_mean, pred_std = npdgp.predict(X_test, S=100)
    mape = mean_absolute_percentage_error(
        to_numpy(y_test), to_numpy(pred_mean)
    )  # , multioutput="raw_values")
    print(f"{mape=}")

    res_dict = batch_assess(
        npdgp,
        X_test,
        y_test,
        y_scale=y_scale,
        device=device,
        S=50,
    )
    wandb.run.summary["baseline_mape"] = baseline_mape
    wandb.run.summary["baseline_rmse"] = baseline_rmse
    wandb.run.summary["baseline_mnll"] = baseline_mnll
    wandb.run.summary["final_mape"] = mape
    wandb.run.summary["final_mnll"] = res_dict["mnll"]
    wandb.run.summary["final_rmse"] = res_dict["rmse"]

    npdgp.plot_output_slices(X_test, save=os.path.join(save_dir, "slice.png"))
    npdgp.plot_features(save=os.path.join(save_dir, "features.png"))
    npdgp.plot_predictions(
        X_test,
        y_test,
        save=os.path.join(save_dir, "preds.png"),
    )

    wandb.log(
        {
            **{
                "slice": wandb.Image(os.path.join(save_dir, "slice.png")),
            },
            **{
                f"amps_{i}": wandb.Image(
                    os.path.join(save_dir, f"features_layer_amps{i}.png")
                )
                for i in range(args.num_layers)
            },
            **{
                f"features_{i}": wandb.Image(
                    os.path.join(save_dir, f"features_layer_{i}.png")
                )
                for i in range(args.num_layers)
            },
        }
    )

    with open(
        os.path.join(save_dir, "optimised_params.txt"), "w", encoding="utf-8"
    ) as f:
        print("Optimised Model Parameters\n", file=f)
        for param_tensor in npdgp.state_dict():
            print(
                param_tensor,
                "\n",
                npdgp.state_dict()[param_tensor].size(),
                "\n",
                npdgp.state_dict()[param_tensor],
                "\n",
                file=f,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1D Toy experiment.")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--output_dir", default="experiment_outputs", type=str)
    parser.add_argument("--num_layers", default=1, type=int)
    parser.add_argument("--n_iter", default=20000, type=int)
    parser.add_argument("--verbosity", default=100, type=int)
    parser.add_argument("--batch_size", default=41, type=int)
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--chunks", default=1, type=int)
    parser.add_argument("--time", default=50.0, type=float)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    run(args)
