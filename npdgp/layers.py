import torch

torch.set_default_dtype(torch.float64)

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from npdgp import integrals
from npdgp.base_gps import FilterGP, InterDomainInputGP
from npdgp.utils import (
    ls2pr,
    pr2ls,
    to_numpy,
    double_integral,
)
from math import pi, sqrt, log


class NPGPLayer(torch.nn.Module):
    """
    Implements a nonparametric multi-output GP layer, which accepts a multivariate input.
    Here, each input/output are assigned their own filter IPs, with the input IPs shared
    across the input dimensions.
    """

    def __init__(
        self,
        d_out,
        init_u_inducing_inputs,
        W=None,
        is_final_layer=False,
        init_filter_width=1.0,
        init_transform_lengthscale=1.5,
        init_amp=1.0,
        init_noise=0.01,
        num_u_functions=None,
        num_filter_points=10,
        device="cpu",
        beta=0.5,
        prior_cov_factor_u=1.0,
        prior_cov_factor_g=0.5,
        prior_mean_factor_g=0.5,
        **kwargs,
    ):
        super(NPGPLayer, self).__init__()
        self.d_out = d_out
        self.d_in = init_u_inducing_inputs.shape[1]
        self.num_u_inducing_points = init_u_inducing_inputs.shape[0]
        self.W = W
        self.is_final_layer = is_final_layer
        self.num_filter_points = num_filter_points
        self.device = device
        if num_u_functions is None:
            num_u_functions = self.d_out
        self.num_u_functions = num_u_functions
        self.init_u_inducing_inputs = init_u_inducing_inputs

        self.register_parameter(
            "noise", torch.nn.Parameter(torch.tensor([init_noise], device=device))
        )

        # Create initial inducing inputs and lengthscales for all filters
        if type(init_filter_width) is list:
            assert len(init_filter_width) == self.d_in
        else:
            init_filter_width = [init_filter_width] * self.d_in

        self.init_g_inducing_inputs = [
            init_filter_width[i]
            * torch.linspace(
                -1, 1, num_filter_points, requires_grad=False, device=device
            ).reshape(-1, 1)
            for i in range(self.d_in)
        ]
        self.init_g_lengthscale = [
            (
                1.0
                * (
                    self.init_g_inducing_inputs[i][1, 0]
                    - self.init_g_inducing_inputs[i][0, 0]
                ).item()
            )
            for i in range(self.d_in)
        ]
        self.init_u_lengthscale = [ls * beta for ls in self.init_g_lengthscale]
        self.init_transform_lengthscale = init_transform_lengthscale

        self.set_gps(
            prior_mean_factor_g, prior_cov_factor_g, prior_cov_factor_u, **kwargs
        )
        try:  # for full
            init_alpha = self.g_gps[0][0].alpha
        except TypeError:  # for fast
            init_alpha = self.g_gps[0].alpha

        normaliser = (2 * init_alpha / pi) ** 0.25
        self.register_parameter(
            "log_amps",
            torch.nn.Parameter(
                log(init_amp)
                + torch.log(normaliser)
                + torch.zeros((self.d_out, self.num_u_functions), device=device)
            ),
        )

    @property
    def amps(self):
        return torch.exp(self.log_amps)

    def set_gps(
        self, prior_mean_factor_g, prior_cov_factor_g, prior_cov_factor_u, **kwargs
    ):
        self.g_gps = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        FilterGP(
                            self.init_g_inducing_inputs[p],
                            init_lengthscale=self.init_g_lengthscale[p],
                            device=self.device,
                            scale_inputs=True,
                            prior_cov_factor=prior_cov_factor_g,
                            prior_mean_factor=prior_mean_factor_g,
                            **kwargs,
                        )
                        for p in range(self.d_in)
                    ]
                )
                for d in range(self.d_out)
            ]
        )
        self.u_gp = InterDomainInputGP(
            self.init_u_inducing_inputs,
            process_lengthscale=self.init_u_lengthscale,
            transform_lengthscale=self.init_transform_lengthscale,
            device=self.device,
            # optimize_inputs=True,
            prior_cov_factor=prior_cov_factor_u,
            num_gps=self.num_u_functions,
            **kwargs,
        )

    def _gs_list2torch(self, d_idx, attr):
        g_gps_d = self.g_gps[d_idx]
        return torch.hstack([getattr(gp, attr) for gp in g_gps_d])

    def _gs_sample_basis(self, d_idx):
        thets = []
        betas = []
        ws = []
        qs = []
        for gp in self.g_gps[d_idx]:
            basis = gp.sample_basis()
            thetd, betasd, wsd = basis
            thets.append(thetd)
            betas.append(betasd)
            ws.append(wsd)
            qs.append(gp.compute_q(basis))
        return (
            torch.hstack(thets),
            torch.swapaxes(torch.stack(betas), 0, 1),
            torch.swapaxes(torch.stack(ws), 0, 1),
            torch.swapaxes(torch.stack(qs), 0, 1),
        )

    def integrate_covariance(self, g_gp, pu, dts, N_tau=100, Ns=5):
        pg = ls2pr(g_gp.lengthscale[0]).item()

        max_tau = max_tau = 3 / sqrt(
            ((g_gp.alpha + 2 * pg) * pu) / (g_gp.alpha + 2 * (pg + pu))
        )
        taus = torch.linspace(-3 * max_tau, 3 * max_tau, N_tau, device=self.device)
        ta, tap = torch.meshgrid((taus, taus))
        kt = torch.zeros(Ns, len(dts), device=self.device)
        for k in range(Ns):
            t = dts[:, None, None]
            tp = torch.zeros((len(dts), 1, 1), device=self.device)
            tau = ta.reshape(1, -1, 1)
            taup = tap.reshape(1, -1, 1)
            norm_part = torch.exp(
                -g_gp.alpha * ((tau) ** 2 + (taup) ** 2)
                - pu * ((t - tau) - (tp - taup)) ** 2
            )
            random_part = g_gp.forward(torch.cat((tau, taup), axis=1).reshape(-1, 1))[0]
            fxy = (
                norm_part[:, :, 0]
                * random_part[None, : tau.shape[1]]
                * random_part[None, tau.shape[1] :]
            ).reshape(-1, N_tau, N_tau)

            kt[k] = double_integral(
                -max_tau, max_tau, -max_tau, max_tau, N_tau, N_tau, fxy
            )
        return kt

    def sample_covariance(self, Ns=5):
        with torch.no_grad():
            out = []
            out_dts = []
            N_tau = 100

            for i, gps in enumerate(self.g_gps):
                outi = []
                out_dtsi = []
                for j, gp in enumerate(gps):
                    u_pr = ls2pr(self.u_gp.process_lengthscale[j]).item()
                    max_a = 1.5 * 3 / sqrt(gp.alpha)
                    max_u = 1.5 * 3 / sqrt(u_pr)
                    max_dt = max(max_a, max_u)
                    dts = torch.linspace(0, max_dt, 100, device=self.device)
                    kt = self.integrate_covariance(gp, u_pr, dts, Ns=Ns)
                    outi.append(kt * torch.sum(self.amps[i]))
                    out_dtsi.append(dts)
                out.append(outi)
                out_dts.append(out_dtsi)
        return (out_dts, out)

    def sample_filter(self, tg=None):
        with torch.no_grad():
            out_ts = []
            out_fs = []
            for i, gps in enumerate(self.g_gps):
                out_tsi = []
                out_fsi = []
                for j, gp in enumerate(gps):
                    max_ip = torch.max(torch.abs(gp.inducing_inputs)).item()
                    tg = torch.linspace(
                        -max_ip, max_ip, 300, requires_grad=False, device=self.device
                    ).reshape(-1, 1)
                    fs = torch.exp(-gp.alpha * tg**2).T * gp.forward(tg)
                    out_fsi.append(fs * torch.sum(self.amps[i]))
                    out_tsi.append(tg)
                out_ts.append(out_tsi)
                out_fs.append(out_fsi)

        return out_ts, out_fs

    def forward(self, ts):

        u_basis = self.u_gp.sample_basis()
        thetaus, betaus, wus = u_basis
        qus = self.u_gp.compute_q(u_basis)
        pus = ls2pr(self.u_gp.cross_lengthscale)
        zus = self.u_gp.inducing_inputs
        thetaus = torch.swapaxes(thetaus, 2, 3)
        ampu = self.u_gp.cross_amp
        out = []

        for i in range(self.d_out):

            pgs = ls2pr(self._gs_list2torch(i, "lengthscale"))
            alphas = self._gs_list2torch(i, "alpha")
            zgs = self._gs_list2torch(i, "inducing_inputs")
            thetags, betags, wgs, qgs = self._gs_sample_basis(i)

            outi = integrals.full_I(
                ts,
                alphas,
                pgs,
                wgs,
                thetags,
                betags,
                zgs,
                qgs,
                pus,
                wus,
                thetaus,
                betaus,
                zus,
                qus,
                ampu,
            )
            out.append(outi)

        layer_out = torch.stack(out, -1)
        layer_out = (self.amps.T[None, :, None, :] * layer_out).sum(axis=1)

        # Apply mean function if not at the final layer
        if self.is_final_layer is False:
            if self.W is None:
                layer_out = layer_out + ts
            else:
                layer_out = layer_out + ts.matmul(self.W)

        return layer_out

    def compute_KL(self):
        kl = 0.0
        for g_d in self.g_gps:
            for g_d_p in g_d:
                kl += g_d_p.compute_KL()

        return kl + self.u_gp.compute_KL()

    def plot_features(self, save=None, covariances=True):

        with torch.no_grad():

            fig = plt.figure(
                constrained_layout=False,
                figsize=(self.d_in * 2, self.d_out * 2),
            )
            grsp = GridSpec(self.d_out, self.d_in, figure=fig)

            if covariances:
                tgs, g_samps = self.sample_covariance()
            else:
                tgs, g_samps = self.sample_filter()
            for j in range(self.d_out):
                for k in range(self.d_in):
                    g_ax = fig.add_subplot(grsp[j, k])
                    gm = torch.mean(g_samps[j][k], axis=0)
                    gs = torch.std(g_samps[j][k], axis=0)

                    ts = to_numpy(tgs[j][k]).flatten()
                    g_ax.plot(
                        ts,
                        to_numpy(gm),
                        c=plt.get_cmap("Set2")(j),
                    )
                    g_ax.fill_between(
                        ts,
                        to_numpy(gm) - to_numpy(gs),
                        to_numpy(gm) + to_numpy(gs),
                        color=plt.get_cmap("Set2")(j),
                        alpha=0.4,
                    )
                    g_ax.plot(
                        ts,
                        to_numpy(g_samps[j][k]).T,
                        color=plt.get_cmap("Set2")(j + 1),
                        alpha=0.2,
                    )
                    if not covariances:
                        g_ax.scatter(
                            to_numpy(self.g_gps[j][k].inducing_inputs),
                            to_numpy(
                                torch.exp(
                                    -self.g_gps[j][k].alpha
                                    * self.g_gps[j][k].inducing_inputs ** 2
                                )[:, 0]
                                * self.g_gps[j][k].variational_dist.variational_mean
                            ),
                            color=plt.get_cmap("Set2")(j),
                            alpha=0.7,
                        )
            plt.tight_layout()
            if save is not None:
                plt.savefig(
                    save,
                    dpi=300,
                    bbox_inches="tight",
                )
            else:
                plt.show()

    def plot_amps(self, save=None):
        fig, ax = plt.subplots()
        arr = to_numpy(self.amps)
        ax.imshow(arr)
        for i in range(self.num_u_functions):
            for j in range(self.d_out):
                text = ax.text(
                    i, j, f"{arr[j, i]:.3f}", ha="center", va="center", color="w"
                )
        ax.set_xlabel("input functions")
        ax.set_ylabel("outputs")
        if save is not None:
            plt.savefig(save)
        else:
            plt.show()

    def objective(self, xs, ys):
        KL = self.compute_KL()
        samps = self.forward(xs)
        like = 0.0
        for i in range(self.d_out):
            y_d = ys[:, i]
            samps_d = samps[:, :, i]
            like += torch.mean(
                torch.distributions.Normal(
                    y_d, self.noise, validate_args=False
                ).log_prob(samps_d)
            )
        return KL - like

    def train(self, data, N_steps, lr):

        pars = dict(self.named_parameters())
        for p in list(pars):
            if "noise" in p:
                pars.pop(p, None)
        opt = torch.optim.Adam(pars.values(), lr=lr)
        for i in range(N_steps):
            opt.zero_grad()
            obj = self.objective(*data)
            obj.backward()
            opt.step()
            if i % 50 == 0:
                print(f"it: {i} obj: {obj}")


class FastNPGPLayer(NPGPLayer):
    def set_gps(
        self, prior_mean_factor_g, prior_cov_factor_g, prior_cov_factor_u, **kwargs
    ):
        self.g_gps = torch.nn.ModuleList(
            [
                FilterGP(
                    self.init_g_inducing_inputs[p],
                    init_lengthscale=self.init_g_lengthscale[p],
                    device=self.device,
                    scale_inputs=True,
                    prior_cov_factor=prior_cov_factor_g,
                    prior_mean_factor=prior_mean_factor_g,
                    **kwargs,
                )
                for p in range(self.d_in)
            ]
        )

        self.u_gp = InterDomainInputGP(
            self.init_u_inducing_inputs,
            process_lengthscale=self.init_u_lengthscale,
            transform_lengthscale=self.init_transform_lengthscale,
            device=self.device,
            prior_cov_factor=prior_cov_factor_u,
            num_gps=self.num_u_functions,
            **kwargs,
        )

    def sample_covariance(self, Ns=5):
        with torch.no_grad():
            out = []
            out_dts = []
            N_tau = 100

            for i, gp in enumerate(self.g_gps):
                u_pr = ls2pr(self.u_gp.process_lengthscale[i]).item()
                max_a = 1.5 * 3 / sqrt(gp.alpha)
                max_u = 1.5 * 3 / sqrt(u_pr)
                max_dt = max(max_a, max_u)
                dts = torch.linspace(0, max_dt, 100, device=self.device)
                kt = self.integrate_covariance(gp, u_pr, dts)
                out.append(kt * torch.mean(self.amps))
                out_dts.append(dts)
        return (out_dts, out)

    def sample_filter(self, tg=None):
        with torch.no_grad():
            out_ts = []
            out_fs = []
            for i, gp in enumerate(self.g_gps):
                max_ip = torch.max(torch.abs(gp.inducing_inputs)).item()
                tg = torch.linspace(
                    -max_ip, max_ip, 300, requires_grad=False, device=self.device
                ).reshape(-1, 1)
                fs = torch.exp(-gp.alpha * tg**2).T * gp.forward(tg)
                out_fs.append(fs * torch.mean(self.amps))
                out_ts.append(tg)

        return out_ts, out_fs

    def forward(self, ts):

        u_basis = self.u_gp.sample_basis()
        thetaus, betaus, wus = u_basis
        qus = self.u_gp.compute_q(u_basis)
        pus = ls2pr(self.u_gp.cross_lengthscale)
        zus = self.u_gp.inducing_inputs
        thetaus = torch.swapaxes(thetaus, 2, 3)
        ampu = self.u_gp.cross_amp

        pgs = ls2pr(torch.hstack([gp.lengthscale for gp in self.g_gps]))
        alphas = torch.hstack([gp.alpha for gp in self.g_gps])
        zgs = torch.hstack([gp.inducing_inputs for gp in self.g_gps])
        bases = [gp.sample_basis() for gp in self.g_gps]
        thetags = torch.hstack([b[0] for b in bases])
        betags = torch.swapaxes(torch.stack([b[1] for b in bases]), 0, 1)
        wgs = torch.swapaxes(torch.stack([b[2] for b in bases]), 0, 1)
        qgs = torch.swapaxes(
            torch.stack(
                [self.g_gps[i].compute_q(bases[i]) for i in range(len(self.g_gps))]
            ),
            0,
            1,
        )

        out = integrals.full_I(
            ts,
            alphas,
            pgs,
            wgs,
            thetags,
            betags,
            zgs,
            qgs,
            pus,
            wus,
            thetaus,
            betaus,
            zus,
            qus,
            ampu,
        )

        layer_out = torch.einsum("oq, bqt -> bto", self.amps, out)
        # Apply mean function if not at the final layer
        if self.is_final_layer is False:
            if self.W is None:
                layer_out = layer_out + ts
            else:
                layer_out = layer_out + ts.matmul(self.W)

        return layer_out

    def compute_KL(self):
        kl = 0.0
        for g_d in self.g_gps:
            kl += g_d.compute_KL()
        return kl + self.u_gp.compute_KL()

    def plot_features(self, save=None, covariances=True):

        with torch.no_grad():

            fig = plt.figure(
                constrained_layout=False,
                figsize=(2, self.d_in * 2),
            )
            grsp = GridSpec(self.d_in, 1, figure=fig)

            if covariances:
                tgs, g_samps = self.sample_covariance()
            else:
                tgs, g_samps = self.sample_filter()

            for k in range(self.d_in):
                g_ax = fig.add_subplot(grsp[k, 0])
                gm = torch.mean(g_samps[k], axis=0)
                gs = torch.std(g_samps[k], axis=0)
                ts = to_numpy(tgs[k]).flatten()
                g_ax.plot(
                    ts,
                    to_numpy(gm),
                    c=plt.get_cmap("Set2")(k),
                )
                g_ax.fill_between(
                    ts,
                    to_numpy(gm) - to_numpy(gs),
                    to_numpy(gm) + to_numpy(gs),
                    color=plt.get_cmap("Set2")(k),
                    alpha=0.4,
                )
                g_ax.plot(
                    ts,
                    to_numpy(g_samps[k]).T,
                    color=plt.get_cmap("Set2")(k + 1),
                    alpha=0.2,
                )
                if not covariances:
                    g_ax.scatter(
                        to_numpy(self.g_gps[k].inducing_inputs),
                        to_numpy(
                            torch.exp(
                                -self.g_gps[k].alpha
                                * self.g_gps[k].inducing_inputs ** 2
                            )[:, 0]
                            * self.g_gps[k].variational_dist.variational_mean
                        ),
                        color=plt.get_cmap("Set2")(k),
                        alpha=0.7,
                    )
            plt.tight_layout()
            if save is not None:
                plt.savefig(
                    save,
                    dpi=300,
                    bbox_inches="tight",
                )
            else:
                plt.show()
