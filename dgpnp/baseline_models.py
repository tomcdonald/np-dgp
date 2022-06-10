import math
import torch
import gpytorch
import tqdm


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    "https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html"

    def __init__(self, inducing_points, num_latents, num_tasks, device):
        # Let's use a different set of inducing points for each latent function
        inducing_points = inducing_points[None, :, :].repeat(num_latents, 1, 1)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        ).to(device)

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        ).to(device)

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents])
        ).to(device)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents]),
        ).to(device)

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_gpytorch_mogp(
    X_train,
    y_train,
    X_valid,
    y_valid,
    inducing_points,
    n_iter=300,
    num_latents=3,
    num_tasks=4,
    device="cpu",
):
    model = MultitaskGPModel(
        inducing_points=inducing_points,
        num_latents=num_latents,
        num_tasks=num_tasks,
        device=device,
    )
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks
    ).to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=0.1,
    )

    # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train.size(0))

    # We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
    # effective for VI.
    iters = tqdm.tqdm(range(n_iter), desc="Epoch")
    print("Fitting GPyTorch model...")
    for i in iters:
        # Within each iteration, we will go over each minibatch of data
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        iters.set_postfix(loss=loss.item())
        loss.backward()
        # print("Epoch %d/%d - Loss: %.3f" % (i + 1, n_iter, loss.item()))
        optimizer.step()

    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Make validation set predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        output = model(X_valid)
        predictions = likelihood(output)
        mnll = -likelihood.log_marginal(y_valid, predictions).mean().item()

    return predictions, mnll
