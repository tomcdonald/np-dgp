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
import json
import pathlib
from datetime import datetime
import wandb


def run(args):

    # Seeds & GPU settings
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("SEED =", SEED)

    str_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    save_dir = os.path.join(args.output_dir, "uci", str_time)

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

    # Load train & test data
    prop = 0.9
    ds = Datasets(data_path="data/")
    data = ds.all_datasets[args.uci_name].get_data(seed=SEED, prop=prop)
    X_train, y_train, X_test, y_test, y_scale = [
        torch.tensor(data[_], dtype=torch.float64, device=device)
        for _ in ["X", "Y", "Xs", "Ys", "Y_std"]
    ]

    # This setting means we just have a train & test set and save the final model;
    # setting True means we have a validation set & save the best model during training.
    save_best_model = False

    if save_best_model:
        # Select fraction of whole dataset to use for validation (data already shuffled)
        valid_prop = 0.1
        n_valid = int(valid_prop * ds.all_datasets[args.uci_name].N)
        X_train, y_train = X_train[n_valid:, :], y_train[n_valid:, :]
        X_valid, y_valid = X_train[:n_valid, :], y_train[:n_valid, :]
    else:
        X_valid, y_valid = X_test, y_test

    # Initialise input proccess IPs & model
    num_u_ips = 100
    u_ip_inits = kmeans_initialisations(num_u_ips, to_numpy(X_train))
    u_ip_inits = torch.tensor(u_ip_inits, device=device)

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
        "save_best_model": save_best_model,
        "fix_noise_iter": 1,
        "model_filepath": os.path.join(save_dir, "model.torch"),
    }

    if args.dry_run:
        wandb_mode = "disabled"
    else:
        wandb_mode = "online"

    config = {
        **train_kwargs,
        **main_kwargs,
        "seed": SEED,
        "dataset": args.uci_name,
    }

    wandb.init(
        project="npdgp-uci",
        entity="npdgp",
        config=config,
        mode=wandb_mode,
    )

    npdgp = NPDeepGP(
        u_inducing_inputs=u_ip_inits, num_outputs=1, X=X_train, **main_kwargs
    ).to(device)

    wandb.watch(npdgp)
    layer_powers = npdgp.assess_layer_power()
    print("Initial internal layer power values:", layer_powers)

    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    npdgp.plot_output_slices(X_train, save=os.path.join(save_dir, "prior_slice.png"))
    npdgp.plot_features(save=os.path.join(save_dir, "prior_features.png"))
    wandb.log(
        {
            "prior_slice": wandb.Image(os.path.join(save_dir, "prior_slice.png")),
        }
    )

    npdgp.train(
        (X_train, y_train),
        (X_valid, y_valid),
        y_scale=y_scale,
        **train_kwargs,
    )
    all_args = {**train_kwargs, **main_kwargs, "seed": SEED}
    layer_powers = npdgp.assess_layer_power()
    print("Final internal layer power values:", layer_powers)

    with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(all_args, f, ensure_ascii=False, indent=4)

    # Load saved model and compute final test set predictions
    npdgp.load_state_dict(torch.load(os.path.join(save_dir, "model.torch")))

    res_dict = batch_assess(
        npdgp,
        X_test,
        y_test,
        y_scale=y_scale,
        device=device,
        S=500,
    )
    wandb.run.summary["final_rmse"] = res_dict["rmse"]
    wandb.run.summary["final_mnll"] = res_dict["mnll"]
    wandb.run.summary["final_powers"] = layer_powers

    with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)

    npdgp.plot_output_slices(X_train, save=os.path.join(save_dir, "slice.png"))
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
                "preds": wandb.Image(os.path.join(save_dir, "preds.png")),
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
    parser = argparse.ArgumentParser(description="UCI experiment.")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--uci_name", default="power", type=str)
    parser.add_argument("--output_dir", default="experiment_outputs", type=str)
    parser.add_argument("--num_layers", default=1, type=int)
    parser.add_argument("--n_iter", default=20000, type=int)
    parser.add_argument("--verbosity", default=100, type=int)
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--chunks", default=1, type=int)
    parser.add_argument("--time", default=50.0, type=float)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    run(args)
