import typing

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from tqdm import tqdm


class TemperedGPR(GaussianProcessRegressor):
    def __init__(self, *, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.current_alpha = alpha

    def log_marginal_likelihood(self, theta, eval_gradient=True, clone_kernel=True, **kwargs):
        res = super().log_marginal_likelihood(
            theta,
            eval_gradient=eval_gradient,
            clone_kernel=clone_kernel,
            **kwargs,
        )
        if eval_gradient:
            lml, grad = res
            return self.current_alpha * lml, self.current_alpha * grad
        return self.current_alpha * res


class BayesianOptimizer:
    def __init__(
        self,
        func,
        kernel,
        bounds: np.ndarray,
        g: typing.Union[np.ndarray, typing.Sequence, float, str],
        alpha: typing.Union[np.ndarray, typing.Sequence, float, str],
        x_init: np.ndarray,
        y_init: np.ndarray,
        md_alpha: float = None,
        xi: float = 0.0,
        n_iter: int = 100,
        acq_samples: int = 100,
        random_state: int = 42,
        hw_clip=(0.01, 1.0),
        hw_eta: float = 0.1,
    ):
        self.func = func
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.n_iter = int(n_iter)
        self.acq_samples = int(acq_samples)
        self.rng = np.random.RandomState(random_state)
        self.xi = float(xi)

        x_init = np.asarray(x_init, dtype=float).reshape(-1, self.dim)
        y_init = np.asarray(y_init, dtype=float).reshape(-1)
        if x_init.shape[0] != y_init.shape[0]:
            raise ValueError("x_init and y_init must have the same number of rows.")

        self.n_input = x_init.shape[0]
        self.end_indx = self.n_input
        self.current_indx = 0
        self.X = np.zeros((self.n_iter + self.n_input, self.dim))
        self.Y = np.zeros((self.n_iter + self.n_input, 1))
        self.X[: self.n_input] = x_init
        self.Y[: self.n_input, 0] = y_init

        self.g = self._coerce_schedule(g, name="g", allow_strings=True)

        self.gp = TemperedGPR(
            alpha=1.0,
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.rng,
        )

        # alpha handling
        self.alpha_type = None
        self.alpha_lb, self.alpha_ub = None, None
        if isinstance(alpha, str):
            self.alpha_lb, self.alpha_ub = 1e-2, 1.0
            if alpha == "hw":
                self.alpha = np.ones(self.n_iter, dtype=float)
            elif "md" in alpha:
                if md_alpha is None:
                    raise ValueError("For 'md' you must pass md_alpha.")
                self.alpha = np.array(
                    [
                        md_alpha / (md_alpha + i)
                        for i in range(self.n_input, self.n_input + self.n_iter)
                    ],
                    dtype=float,
                )
            else:
                raise ValueError("Unrecognized alpha argument!")
            self.alpha_type = alpha
        else:
            self.alpha = self._coerce_schedule(alpha, name="alpha", allow_strings=False).astype(float)

        self.gp.current_alpha = float(self.alpha[0])
        self._signal_kernel = None
        self._noise_var = None            # standardized-y scale
        self._noise_var_orig = None       # original-y scale

        # prequential stats for HW estimator
        self._hw_eps = 1e-12
        self._hw_eta = float(hw_eta)
        self._hw_clip = tuple(hw_clip)
        self._sum_num = self._hw_eps
        self._sum_den = self._hw_eps
        self._noise_hat2 = None           # original-y scale

        # per-iteration caches
        self._posterior_cache = {}
        self._incumbent_cache = {}
        self._mes_cache = {}

    def _coerce_schedule(self, values, name="schedule", allow_strings=False):
        if isinstance(values, str):
            if not allow_strings:
                raise ValueError(f"{name} cannot be a string here.")
            return np.full(self.n_iter, values, dtype=object)

        if np.isscalar(values):
            return np.full(self.n_iter, values, dtype=object if allow_strings else float)

        arr = np.asarray(values, dtype=object if allow_strings else float).reshape(-1)
        if arr.size == 1:
            return np.full(self.n_iter, arr[0], dtype=arr.dtype)
        if arr.size != self.n_iter:
            raise ValueError(f"{name} must have length n_iter={self.n_iter}, got {arr.size}.")
        return arr.copy()

    def _invalidate_caches(self):
        self._posterior_cache.clear()
        self._incumbent_cache.clear()
        self._mes_cache.clear()

    def _cache_key(self, cur_alpha: float):
        return self.end_indx, round(float(cur_alpha), 12)

    def _alpha_at(self, idx: int) -> float:
        idx = int(np.clip(idx, 0, len(self.alpha) - 1))
        return float(self.alpha[idx])

    def _current_g(self):
        return self.g[self.current_indx]

    def get_hw_alpha(self):
        """
        Holmes–Walker empirical alpha (global) via the prequential method-of-moments:
        alpha_t = sqrt( avg_{s<=t-1}(PV_s + hat_sigma^2_s) / avg_{s<=t-1}(PV_s + MSE_s) ),
        clipped to hw_clip.

        The implementation uses one-step-ahead *untempered* predictive moments and keeps
        the numerator / denominator on the original response scale, matching the paper.
        """
        if self.current_indx == 0 or self._sum_den <= 0:
            return 1.0
        ratio = max(self._sum_num / self._sum_den, self._hw_eps)
        alpha = float(np.sqrt(ratio))
        return float(np.clip(alpha, self._hw_clip[0], self._hw_clip[1]))

    def update_model(self, fit_alpha: float = None):
        """Refit the GP to all observed data and cache kernel pieces."""
        if self.end_indx <= 0:
            raise ValueError("Need at least one observed point to fit the GP.")

        if fit_alpha is None:
            fit_alpha = self._alpha_at(self.current_indx)
        self.gp.current_alpha = float(fit_alpha)

        X_train = self.X[: self.end_indx]
        y_train = self.Y[: self.end_indx].ravel()
        self.gp.fit(X_train, y_train)

        summed = self.gp.kernel_
        self._signal_kernel = summed.k1
        self._noise_var = max(float(summed.k2.noise_level), self._hw_eps)

        y_std = getattr(self.gp, "_y_train_std", 1.0)
        if np.isscalar(y_std):
            y_std_arr = float(y_std)
        else:
            y_std_arr = float(np.asarray(y_std).reshape(-1)[0])
        if y_std_arr == 0:
            y_std_arr = 1.0
        self._noise_var_orig = max(self._noise_var * (y_std_arr ** 2), self._hw_eps)

        if self._noise_hat2 is None:
            self._noise_hat2 = self._noise_var_orig

        self._invalidate_caches()

    def _ensure_2d(self, x):
        arr = np.asarray(x, dtype=float)
        return arr.reshape(-1, self.dim)

    def _get_posterior_state(self, cur_alpha):
        key = self._cache_key(cur_alpha)
        if key in self._posterior_cache:
            return self._posterior_cache[key]

        X_train = self.X[: self.end_indx]
        y_train = self.Y[: self.end_indx].ravel()

        if getattr(self.gp, "normalize_y", False):
            y_mean = getattr(self.gp, "_y_train_mean", 0.0)
            y_std = getattr(self.gp, "_y_train_std", 1.0)
            if np.isscalar(y_std):
                y_std = float(y_std)
            else:
                y_std = float(np.asarray(y_std).reshape(-1)[0])
            if y_std == 0:
                y_std = 1.0
            y_tilde = (y_train - y_mean) / y_std
        else:
            y_mean, y_std = 0.0, 1.0
            y_tilde = y_train

        K_tt = self._signal_kernel(X_train, X_train)
        eff_noise = max(self._noise_var / float(cur_alpha), self._hw_eps)
        L, lower = cho_factor(K_tt + eff_noise * np.eye(len(X_train)), lower=True)
        alpha_vec = cho_solve((L, lower), y_tilde)

        state = {
            "X_train": X_train,
            "y_mean": y_mean,
            "y_std": y_std,
            "L": L,
            "lower": lower,
            "alpha_vec": alpha_vec,
        }
        self._posterior_cache[key] = state
        return state

    def predict_tempered(self, X, cur_alpha):
        """
        Posterior under tempered likelihood: equivalent to using effective noise var = noise / alpha.
        When normalize_y=True in sklearn GPR, the algebra is carried out in standardized y-space
        and then mapped back to the original response scale.
        """
        X = self._ensure_2d(X)
        state = self._get_posterior_state(cur_alpha)
        X_train = state["X_train"]
        K_ts = self._signal_kernel(X_train, X)
        K_ss = self._signal_kernel(X, X)

        mu_tilde = K_ts.T.dot(state["alpha_vec"])
        K_inv_Kts = cho_solve((state["L"], state["lower"]), K_ts)
        cov_tilde = K_ss - K_ts.T.dot(K_inv_Kts)
        std_tilde = np.sqrt(np.maximum(np.diag(cov_tilde), 0.0))

        mu = mu_tilde * state["y_std"] + state["y_mean"]
        std = std_tilde * state["y_std"]
        return mu.reshape(-1, 1), std.reshape(-1, 1)

    def _get_incumbent_mean(self, cur_alpha):
        key = self._cache_key(cur_alpha)
        if key not in self._incumbent_cache:
            mu_train, _ = self.predict_tempered(self.X[: self.end_indx], cur_alpha)
            self._incumbent_cache[key] = float(np.max(mu_train))
        return self._incumbent_cache[key]

    def probability_improvement(self, x, cur_alpha):
        x = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_sample_opt = self._get_incumbent_mean(cur_alpha)
        imp = mu - mu_sample_opt - self.xi
        Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=(sigma > 0))
        vals = norm.cdf(Z).ravel()
        return float(vals[0]) if x.shape[0] == 1 else vals

    def expected_improvement(self, x, cur_alpha):
        x = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_sample_opt = self._get_incumbent_mean(cur_alpha)
        imp = mu - mu_sample_opt - self.xi
        Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=(sigma > 0))
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = np.where(sigma <= 0.0, 0.0, ei)
        vals = ei.ravel()
        return float(vals[0]) if x.shape[0] == 1 else vals

    def power_2_expected_improvement(self, x, cur_alpha):
        x = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_sample_opt = self._get_incumbent_mean(cur_alpha)
        imp = mu - mu_sample_opt - self.xi
        Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=(sigma > 0))
        result = sigma ** 2 * (norm.cdf(Z) * (1.0 + Z ** 2) + Z * norm.pdf(Z))
        result = np.where(sigma <= 0.0, 0.0, result)
        vals = result.ravel()
        return float(vals[0]) if x.shape[0] == 1 else vals

    def power_3_expected_improvement(self, x, cur_alpha):
        x = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_sample_opt = self._get_incumbent_mean(cur_alpha)
        imp = mu - mu_sample_opt - self.xi
        Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=(sigma > 0))
        result = sigma ** 3 * (norm.cdf(Z) * (3.0 * Z + Z ** 3) + (Z ** 2 + 2.0) * norm.pdf(Z))
        result = np.where(sigma <= 0.0, 0.0, result)
        vals = result.ravel()
        return float(vals[0]) if x.shape[0] == 1 else vals

    def _get_mes_ystar_samples(self, cur_alpha, n_ystar=32, representer_X=None):
        if representer_X is None:
            representer_X = self.X[: self.end_indx]
        representer_X = self._ensure_2d(representer_X)
        key = (self._cache_key(cur_alpha), int(n_ystar), representer_X.shape[0])
        if key in self._mes_cache:
            return self._mes_cache[key]

        mu_R, std_R = self.predict_tempered(representer_X, cur_alpha)
        mu_R = mu_R.ravel()
        std_R = std_R.ravel()
        draws = self.rng.standard_normal(size=(int(n_ystar), mu_R.shape[0]))
        ystar_samples = np.max(mu_R[None, :] + std_R[None, :] * draws, axis=1)
        self._mes_cache[key] = ystar_samples
        return ystar_samples

    def max_value_entropy_search(
        self,
        x,
        cur_alpha,
        ystar_samples=None,
        n_ystar=32,
        representer_X=None,
        random_state=None,
    ):
        """
        MES acquisition (noiseless/latent variant) with numerically stable eta computation.
        Returns scalar if x is (d,), else a 1D array (n,).
        """
        _ = random_state  # kept for API compatibility

        X = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(X, cur_alpha)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        safe = sigma > 0
        if not np.any(safe):
            return 0.0 if X.shape[0] == 1 else np.zeros(X.shape[0])

        if ystar_samples is None:
            ystar_samples = self._get_mes_ystar_samples(
                cur_alpha,
                n_ystar=n_ystar,
                representer_X=representer_X,
            )
        else:
            ystar_samples = np.asarray(ystar_samples, dtype=float).ravel()
            if ystar_samples.size == 0:
                return 0.0 if X.shape[0] == 1 else np.zeros(X.shape[0])

        gamma = (ystar_samples[None, :] - mu) / np.maximum(sigma, 1e-12)
        logPhi = log_ndtr(gamma)
        logphi = -0.5 * gamma ** 2 - 0.5 * np.log(2.0 * np.pi)

        eta = np.empty_like(gamma)
        tail_mask = logPhi < -20.0
        eta[~tail_mask] = np.exp(logphi[~tail_mask] - logPhi[~tail_mask])

        gm = gamma[tail_mask]
        invg = 1.0 / np.where(gm != 0.0, gm, -1e-12)
        eta[tail_mask] = -gm - invg + 2.0 * invg ** 3

        per_sample = 0.5 * gamma * eta - logPhi
        mes_vals = np.mean(per_sample, axis=1)
        mes_vals = np.where(safe.ravel(), mes_vals, 0.0)
        mes_vals = np.nan_to_num(mes_vals, neginf=0.0, posinf=1e6)
        return float(mes_vals[0]) if X.shape[0] == 1 else mes_vals

    def general_expected_improvement(self, x, cur_alpha):
        """General tempered g-EI for arbitrary nonnegative g using numerical integration."""
        x = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu = mu.ravel()
        sigma = sigma.ravel()
        mu_sample_opt = self._get_incumbent_mean(cur_alpha)
        g_cur = float(self._current_g())

        vals = np.zeros(x.shape[0], dtype=float)
        for i, (mu_i, sigma_i) in enumerate(zip(mu, sigma)):
            if sigma_i <= 0:
                vals[i] = 0.0
                continue
            v = (mu_sample_opt + self.xi - mu_i) / sigma_i
            integrand = lambda u: (u - v) ** g_cur * norm.pdf(u)
            val, _ = quad(integrand, v, np.inf, limit=100)
            vals[i] = (sigma_i ** g_cur) * val
        return float(vals[0]) if x.shape[0] == 1 else vals

    def propose_location(self, temp_alpha=None):
        """Sample candidates and pick the one with maximal acquisition."""
        if temp_alpha is None:
            temp_alpha = self.get_hw_alpha() if self.alpha_type == "hw" else self._alpha_at(self.current_indx)
        temp_alpha = float(temp_alpha)

        best_x, best_val = None, np.inf
        X_cand = self.rng.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(self.acq_samples, self.dim),
        )

        g_cur = self._current_g()
        if g_cur == 1:
            acq = lambda z: -self.expected_improvement(z, temp_alpha)
        elif g_cur == 0:
            acq = lambda z: -self.probability_improvement(z, temp_alpha)
        elif g_cur == 2:
            acq = lambda z: -self.power_2_expected_improvement(z, temp_alpha)
        elif g_cur == 3:
            acq = lambda z: -self.power_3_expected_improvement(z, temp_alpha)
        elif g_cur == "MES":
            ystar_samples = self._get_mes_ystar_samples(temp_alpha)
            acq = lambda z: -self.max_value_entropy_search(z, temp_alpha, ystar_samples=ystar_samples)
        else:
            acq = lambda z: -self.general_expected_improvement(z, temp_alpha)

        for x0 in X_cand:
            res = minimize(acq, x0=x0, bounds=self.bounds.tolist(), method="L-BFGS-B")
            if res.fun < best_val:
                best_val = float(res.fun)
                best_x = res.x.copy()
        return best_x, temp_alpha

    def visualize(self, iteration_idx, data_end_idx):
        plt.figure(figsize=(8, 4))
        x_star = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 500)
        f_star = self.func(x_star)
        plt.plot(x_star, f_star, "k-", label="true f(x)", alpha=0.5)
        data_x, data_y = self.X[:data_end_idx], self.Y[:data_end_idx]
        plt.scatter(data_x, data_y, c="C1", s=50, edgecolor="k")
        for i, (x, y) in enumerate(zip(data_x[self.n_input :], data_y[self.n_input :])):
            plt.text(x, y, str(i), color="black", fontsize=9, ha="center", va="bottom")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(f"iter {iteration_idx}: alpha={self.alpha[iteration_idx]}, g={self.g[iteration_idx]}")

        mu, std = self.predict_tempered(x_star.reshape(-1, self.dim), self.alpha[iteration_idx])
        plt.plot(x_star, mu, "b--", label="GP mean", alpha=0.5)
        plt.fill_between(
            x_star.ravel(),
            (mu - std).ravel(),
            (mu + std).ravel(),
            color="blue",
            alpha=0.2,
            label=r"$\pm1\,\sigma$",
        )

        if self.g[iteration_idx] == 1:
            acq_y = self.expected_improvement(x_star.reshape(-1, self.dim), self.alpha[iteration_idx])
            plt.plot(x_star, acq_y, linestyle="--", color="C1", label="EI")
        plt.legend()
        plt.show()

    def _draw_observation_noise(self, x, y_latent, noise_std, noise_sampler=None, iteration=None):
        """Draw observation noise for a single candidate point.

        If ``noise_sampler`` is provided, it should accept ``X`` with shape ``(n, d)`` and
        may also accept ``rng``, ``y_latent``, ``iteration``, and ``optimizer`` keyword
        arguments. It must return one noise draw per row of ``X``.
        """
        if noise_sampler is None:
            return float(self.rng.normal(0.0, noise_std))

        X = np.asarray(x, dtype=float).reshape(1, -1)
        sampled = noise_sampler(
            X,
            rng=self.rng,
            y_latent=np.asarray([y_latent], dtype=float),
            iteration=iteration,
            optimizer=self,
        )
        sampled = np.asarray(sampled, dtype=float).reshape(-1)
        if sampled.size != 1:
            raise ValueError(
                f"noise_sampler must return one draw for a single x; got shape {sampled.shape}."
            )
        return float(sampled[0])

    def simulate_optimization(
        self,
        noise_std=0.01,
        noise_sampler=None,
        visualize=False,
        max_val=None,
        verbose=True,
    ):
        if self.n_input > 0:
            self.update_model(fit_alpha=self._alpha_at(0))

        inst_regrets = np.zeros(self.n_iter, dtype=float) if max_val is not None else None
        noise_history = np.zeros(self.n_iter, dtype=float)
        iterator = tqdm(range(self.n_iter), disable=not verbose)

        for i in iterator:
            x_next, temp_alpha = self.propose_location()

            # One-step-ahead *untempered* predictive moments for the HW schedule.
            mu_pre, std_pre = self.predict_tempered(np.asarray(x_next).reshape(1, -1), 1.0)
            mu_pre = float(mu_pre.ravel()[0])
            var_pre = float(std_pre.ravel()[0] ** 2)

            y_latent = float(np.asarray(self.func(np.asarray(x_next).reshape(1, -1))).ravel()[0])
            noise = self._draw_observation_noise(
                x_next,
                y_latent=y_latent,
                noise_std=noise_std,
                noise_sampler=noise_sampler,
                iteration=i,
            )
            noise_history[i] = float(noise)
            y_next = y_latent + noise

            if max_val is not None:
                inst_regrets[i] = float(max_val - y_latent)

            resid2 = float((y_next - mu_pre) ** 2)
            noise_prev = float(
                self._noise_hat2
                if self._noise_hat2 is not None
                else (self._noise_var_orig if self._noise_var_orig is not None else max(noise_std ** 2, self._hw_eps))
            )
            self._sum_num += var_pre + max(noise_prev, self._hw_eps)
            self._sum_den += var_pre + resid2
            tilde = max(0.0, resid2 - var_pre)
            self._noise_hat2 = (1.0 - self._hw_eta) * noise_prev + self._hw_eta * tilde

            self.X[self.end_indx] = x_next
            self.Y[self.end_indx, 0] = y_next
            if self.alpha_type == "hw":
                self.alpha[self.current_indx] = temp_alpha


            self.end_indx += 1
            self.current_indx += 1
            self.update_model(fit_alpha=temp_alpha)


            if visualize:
                if self.dim > 1:
                    raise NotImplementedError("Only 1D visualize implemented")
                self.visualize(self.current_indx - 1, self.end_indx)

            if verbose:
                iterator.set_postfix(
                    alpha=f"{temp_alpha:.3f}",
                    y=f"{y_next:.4f}",
                    best_obs=f"{self.Y[:self.end_indx].max():.4f}",
                )

        X_eval = self.X[: self.end_indx].copy()
        Y_observed = self.Y[: self.end_indx].ravel().copy()
        Y_latent = np.asarray(self.func(X_eval)).ravel().copy()

        best_observed_trajectory = np.maximum.accumulate(Y_observed)
        best_latent_trajectory = np.maximum.accumulate(Y_latent)
        observed_gap_trajectory = None if max_val is None else (max_val - best_observed_trajectory)
        simple_regret_trajectory = None if max_val is None else (max_val - best_latent_trajectory)
        cum_regrets = None if inst_regrets is None else np.cumsum(inst_regrets)

        best_observed_idx = int(np.argmax(Y_observed))
        best_latent_idx = int(np.argmax(Y_latent))
        best_regret_idx = int(np.argmin(inst_regrets)) if inst_regrets is not None else None

        return {
            "best_iteration_observed": best_observed_idx - self.n_input,
            "best_iteration_observed_all": best_observed_idx,
            "best_observed": float(best_observed_trajectory[-1]),
            "best_observed_x": X_eval[best_observed_idx].copy(),
            "best_iteration_latent": best_latent_idx - self.n_input,
            "best_iteration_latent_all": best_latent_idx,
            "best_latent": float(best_latent_trajectory[-1]),
            "best_latent_x": X_eval[best_latent_idx].copy(),
            "best_iteration_regret": best_regret_idx,
            "best_observed_trajectory": best_observed_trajectory,
            "best_latent_trajectory": best_latent_trajectory,
            "observed_gap_trajectory": observed_gap_trajectory,
            "simple_regret_trajectory": simple_regret_trajectory,
            "inst_regrets": inst_regrets,
            "cum_regrets": cum_regrets,
            "alpha_history": self.alpha[: self.current_indx].copy(),
            "X_evaluated": X_eval,
            "Y_observed": Y_observed,
            "Y_latent": Y_latent,
            "noise_history": noise_history,
        }
