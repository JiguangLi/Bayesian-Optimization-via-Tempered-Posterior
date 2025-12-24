import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from scipy.special import log_ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Matern,
    WhiteKernel,
)
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import typing




class TemperedGPR(GaussianProcessRegressor):
    def __init__(self, *, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.current_alpha = alpha

    def log_marginal_likelihood(self, theta, eval_gradient=True, clone_kernel=True, **kwargs):
        res = super().log_marginal_likelihood(theta, eval_gradient=eval_gradient,
                                              clone_kernel=clone_kernel, **kwargs)
        if eval_gradient:
            lml, grad = res
            return self.current_alpha * lml, self.current_alpha * grad
        else:
            return self.current_alpha * res


class BayesianOptimizer:
    def __init__(
        self,
        func,
        kernel,
        bounds: np.ndarray,
        g: np.ndarray,
        alpha: typing.Union[np.ndarray, str],
        x_init: np.ndarray,
        y_init: np.ndarray,
        md_alpha: float = None,
        xi: float = 0.0,
        n_iter=100,
        acq_samples=100,
        random_state=42,
        hw_clip=(0.01, 1.0),       
        hw_eta=0.1               
    ):
        self.func = func
        self.bounds = bounds
        self.dim = self.bounds.shape[0]
        self.g = g
        self.n_iter = n_iter
        self.acq_samples = acq_samples
        self.rng = np.random.RandomState(random_state)
        self.n_input = x_init.shape[0]
        self.end_indx = self.n_input
        self.current_indx = 0
        self.X = np.zeros((n_iter + self.n_input, self.dim))
        self.Y = np.zeros((n_iter + self.n_input, 1))
        self.X[:self.n_input] = x_init
        self.Y[:self.n_input] = y_init.reshape(-1, 1)
        self.xi = xi

        self.gp = TemperedGPR(
            alpha=1.0,
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.rng,
        )

        # alpha handling
        if isinstance(alpha, np.ndarray):
            self.alpha = alpha
            self.alpha_type = None
            self.alpha_lb, self.alpha_ub = None, None
        else:
            self.alpha_lb, self.alpha_ub = 1e-2, 1.0
            if alpha == "hw":
                self.alpha = np.ones(self.n_iter)
            elif "md" in alpha:
                if md_alpha is None:
                    raise ValueError("For 'md' you must pass md_alpha.")
                self.alpha = np.array([md_alpha/(md_alpha+i)
                                       for i in range(self.n_input, self.n_input+n_iter)])
            else:
                raise ValueError("Unrecognized alpha argument!")
            self.alpha_type = alpha

        self.gp.current_alpha = float(self.alpha[0])
        self._signal_kernel = None
        self._noise_var = None

        # prequential stats for HW estimator
        self._hw_eps = 1e-12
        self._hw_eta = float(hw_eta)
        self._hw_clip = tuple(hw_clip)
        self._sum_num = self._hw_eps   # sum of (sigma_{t-1}^2(x_t) + hat_sigma_{t-1}^2)
        self._sum_den = self._hw_eps   # sum of (sigma_{t-1}^2(x_t) + (y_t - mu_{t-1}(x_t))^2)
        self._noise_hat2 = None        # EWMA noise estimate (prequential)

    def get_hw_alpha(self):
        """
        Holmes–Walker empirical alpha (global) via prequential method-of-moments:
        Returns current alpha to use for PROPOSAL at this iteration (based on history up to t-1).
        """
        # If no prequential data yet, return 1
        if self.current_indx == 0 or self._sum_den <= 0:
            return 1.0
        ratio = self._sum_num / self._sum_den
        # Numerical safety
        ratio = max(ratio, self._hw_eps)
        alpha2 = ratio
        alpha = float(np.sqrt(alpha2))
        alpha = float(np.clip(alpha, self._hw_clip[0], self._hw_clip[1]))
        return alpha

    def update_model(self):
        """Refit the GP to all observed data and cache kernel pieces."""
        # Note: hyperparam fit is unaffected by alpha scaling (argmax invariant)
        self.gp.current_alpha = float(self.alpha[self.current_indx])
        X_train = self.X[: self.end_indx]
        y_train = self.Y[: self.end_indx].ravel()
        self.gp.fit(X_train, y_train)
        summed = self.gp.kernel_
        # assume: kernel = (signal) + WhiteKernel(noise)
        self._signal_kernel = summed.k1
        self._noise_var = float(summed.k2.noise_level)
        # initialize EWMA noise if not set (use fitted noise as starting point)
        if self._noise_hat2 is None:
            self._noise_hat2 = max(self._noise_var, self._hw_eps)

    def _ensure_2d(self, x):
          # x might be scalar, 1‑D, or already 2‑D
        arr = np.asarray(x)
        return arr.reshape(-1, self.dim)

    def predict_tempered(self, X, cur_alpha):
        """
        Posterior under tempered likelihood: equivalent to using effective noise var = noise/alpha.
        Note: handle sklearn's normalize_y=True by working in standardized y-space and mapping back.
        """
        X = np.asarray(X).reshape(-1, self.dim)
        X_train = self.X[: self.end_indx]
        y_train = self.Y[: self.end_indx].ravel()

        # y normalization as in sklearn GPR
        if getattr(self.gp, "normalize_y", False):
            y_mean = getattr(self.gp, "_y_train_mean", 0.0)
            y_std = getattr(self.gp, "_y_train_std", 1.0)
            if y_std == 0:
                y_std = 1.0
            y_tilde = (y_train - y_mean) / y_std
        else:
            y_mean, y_std = 0.0, 1.0
            y_tilde = y_train

        # Covariances using optimized hyperparams
        K_tt = self._signal_kernel(X_train, X_train)
        K_ts = self._signal_kernel(X_train, X)
        K_ss = self._signal_kernel(X, X)

        # Tempered noise in standardized space (scaling commutes)
        eff_noise = (self._noise_var / cur_alpha)
        # Cholesky solve
        L, lower = cho_factor(K_tt + eff_noise * np.eye(len(X_train)), lower=True)
        alpha_vec = cho_solve((L, lower), y_tilde)
        mu_tilde = K_ts.T.dot(alpha_vec)
        K_inv_Kts = cho_solve((L, lower), K_ts)
        cov_tilde = K_ss - K_ts.T.dot(K_inv_Kts)
        std_tilde = np.sqrt(np.maximum(np.diag(cov_tilde), 0.0))

        # map back to original scale
        mu = mu_tilde * y_std + y_mean
        std = std_tilde * y_std
        return mu.reshape(-1, 1), std.reshape(-1, 1)

    def probability_improvement(self, x, cur_alpha):
        """Standard Probablity Improvement with Alpha Posterior Temperaing"""
        x = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_train, _ = self.predict_tempered(self.X[:self.end_indx], cur_alpha)
        mu_sample_opt = np.max(mu_train)
        imp = mu - mu_sample_opt - self.xi
        Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=(sigma > 0))
        return norm.cdf(Z).ravel()[0] if mu.shape[0] == 1 else norm.cdf(Z).ravel()

    def expected_improvement(self, x, cur_alpha):
        """Standard Expected Improvement with Alpha Posterior Temperaing"""
        x = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_train, _ = self.predict_tempered(self.X[:self.end_indx], cur_alpha)
        mu_sample_opt = np.max(mu_train)
        imp = mu - mu_sample_opt - self.xi
        Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=(sigma > 0))
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = np.where(sigma <= 0.0, 0.0, ei)
        return ei.ravel()[0] if mu.shape[0] == 1 else ei.ravel()

    def power_2_expected_improvement(self, x, cur_alpha):
        """generalized expected improvement when g=2"""
        x = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_train, _ = self.predict_tempered(self.X[:self.end_indx], cur_alpha)
        mu_sample_opt = np.max(mu_train)
        imp = mu - mu_sample_opt - self.xi
        Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=(sigma > 0))
        result = sigma**2 * (norm.cdf(Z) * (1 + Z**2) + Z * norm.pdf(Z))
        return result.ravel()[0] if mu.shape[0] == 1 else result.ravel()

    def power_3_expected_improvement(self, x, cur_alpha):
        """generalized expected improvement when g=3"""
        x = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_train, _ = self.predict_tempered(self.X[:self.end_indx], cur_alpha)
        mu_sample_opt = np.max(mu_train)
        imp = mu - mu_sample_opt - self.xi
        Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=(sigma > 0))
        result = sigma**3 * (norm.cdf(Z) * (3*Z + Z**3) + (Z**2 + 2) * norm.pdf(Z))
        return result.ravel()[0] if mu.shape[0] == 1 else result.ravel()
    
    def max_value_entropy_search(self, x, cur_alpha,
                             ystar_samples=None,
                             n_ystar=32,
                             representer_X=None,
                             random_state=None):
        """
        MES acquisition (noiseless/latent variant) with numerically stable eta computation.
        Returns scalar if x is (d,), else a 1D array (n,).
        """
        rng = (random_state if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state))

        X = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(X, cur_alpha)  # (n,1) each
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # If predictive std ~ 0, MES should be 0 there.
        safe = (sigma > 0)
        if not np.any(safe):
            return 0.0 if X.shape[0] == 1 else np.zeros(X.shape[0])

        # ---------------- y* sampling (once per BO iteration, ideally cached) ----------------
        if ystar_samples is None:
            R = representer_X if representer_X is not None else self.X[: self.end_indx]
            mu_R, std_R = self.predict_tempered(R, cur_alpha)  # (R,1) each
            mu_R = mu_R.ravel()
            std_R = std_R.ravel()

            ystar = np.empty(n_ystar, dtype=float)
            for m in range(n_ystar):
                z = rng.standard_normal(size=mu_R.shape[0])
                ydraw = mu_R + std_R * z
                ystar[m] = np.max(ydraw)
            ystar_samples = ystar
        else:
            ystar_samples = np.asarray(ystar_samples, dtype=float).ravel()
            if ystar_samples.size == 0:
                return 0.0 if X.shape[0] == 1 else np.zeros(X.shape[0])

        # ---------------- stable per-sample MES term ----------------
        # gamma shape: (n, M)
        gamma = (ystar_samples[None, :] - mu) / np.maximum(sigma, 1e-12)

        # tail-stable log Phi (no -inf for large negative gamma)
        logPhi = log_ndtr(gamma)
        # log phi = -0.5*gamma^2 - 0.5*log(2π)
        logphi = -0.5*gamma**2 - 0.5*np.log(2.0*np.pi)

        # Compute eta = phi/Phi safely:
        #  - in the "safe" region (Phi not tiny), use exp(logphi - logPhi)
        #  - in the far left tail, use Mills-ratio asymptotic: eta ≈ -γ - 1/γ
        eta = np.empty_like(gamma)
        # threshold: Phi < ~e^-20 ≈ 2e-9  -> use asymptotic
        tail_mask = (logPhi < -20.0)

        eta[~tail_mask] = np.exp(logphi[~tail_mask] - logPhi[~tail_mask])

        gm = gamma[tail_mask]
        invg = 1.0 / np.where(gm != 0.0, gm, -1e-12)
        eta[tail_mask] = -gm - invg  + 2*invg**3 

        # per-sample MES: 0.5*gamma*eta - log Phi
        per_sample = 0.5 * gamma * eta - logPhi  # (n, M)

        mes_vals = np.mean(per_sample, axis=1)   # average over y* samples -> (n,)

        # zero where predictive variance ~ 0 and clean NaNs/Infs
        mes_vals = np.where(safe.ravel(), mes_vals, 0.0)
        mes_vals = np.nan_to_num(mes_vals, neginf=0.0, posinf=1e6)

        return mes_vals[0] if X.shape[0] == 1 else mes_vals



    def general_expected_improvement(self, x, cur_alpha):
        """General Expected Improvement with Alpha Posterior Temperaing
        
        Note currently only accept 1 input
        """
        x = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_train, _ = self.predict_tempered(self.X[:self.end_indx], cur_alpha)
        mu_sample_opt = np.max(mu_train)
        v = ((mu_sample_opt - mu) / sigma).item()
        integrand = lambda u: (u - v)**self.g[self.current_indx] * norm.pdf(u)
        val, _ = quad(integrand, v, np.inf, limit=100)
        return (sigma**self.g[self.current_indx]).ravel()[0] * val

    def propose_location(self, temp_alpha=None):
        """Sample candidates and pick the one with max acquisition."""
        # Compute alpha once per iteration if needed
        if temp_alpha is None:
            if self.alpha_type == "hw":
                temp_alpha = self.get_hw_alpha()
            else:
                temp_alpha = float(self.alpha[self.current_indx])

        best_x, best_val = None, np.inf
        X_cand = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                  size=(self.acq_samples, self.dim))

        # choose acquisition
        if self.g[self.current_indx] == 1:
            acq = lambda z: -self.expected_improvement(z, temp_alpha)
        elif self.g[self.current_indx] == 0:
            acq = lambda z: -self.probability_improvement(z, temp_alpha)
        elif self.g[self.current_indx] == 2:
            acq = lambda z: -self.power_2_expected_improvement(z, temp_alpha)
        elif self.g[self.current_indx] == 3:
            acq = lambda z: -self.power_3_expected_improvement(z, temp_alpha)
        elif self.g[self.current_indx] == "MES":
            acq = lambda z: -self.max_value_entropy_search(z, temp_alpha)
        else:
            acq = lambda z: -self.general_expected_improvement(z, temp_alpha)

        for x in X_cand:
            res = minimize(acq, x0=x, bounds=self.bounds.tolist(), method="L-BFGS-B")
            if res.fun < best_val:
                best_val, best_x = res.fun, res.x
        return best_x, float(temp_alpha)

    def visualize(self, iteration_idx, data_end_idx):
        plt.figure(figsize=(8, 4))
        x_star = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 500)
        f_star = self.func(x_star)
        plt.plot(x_star, f_star, 'k-', label='true f(x)', alpha=0.5)
        data_x, data_y = self.X[:data_end_idx], self.Y[:data_end_idx]
        plt.scatter(data_x, data_y, c='C1', s=50, edgecolor='k')
        for i, (x, y) in enumerate(zip(data_x[self.n_input:], data_y[self.n_input:])):
            plt.text(x, y, str(i), color='black', fontsize=9, ha='center', va='bottom')
        plt.xlabel('x'); plt.ylabel('f(x)')
        plt.title(f"iter {iteration_idx}: alpha={self.alpha[iteration_idx]}, g={self.g[iteration_idx]}")

        # posterior mean/uncertainty under current alpha
        mu, std = self.predict_tempered(x_star.reshape(-1, self.dim), self.alpha[iteration_idx])
        plt.plot(x_star, mu, 'b--', label='GP mean', alpha=0.5)
        plt.fill_between(x_star.ravel(), (mu - std).ravel(), (mu + std).ravel(),
                         color='blue', alpha=0.2, label=r'$\pm1\,\sigma$')

        if self.g[iteration_idx] == 1:
            acq_y = self.expected_improvement(x_star.reshape(-1, self.dim), self.alpha[iteration_idx])
            plt.plot(x_star, acq_y, linestyle='--', color='C1', label='EI')
        plt.legend(); plt.show()

    def simulate_optimization(self, noise_std=0.01, visualize=False, max_val=None, verbose=True):
        if self.n_input > 0:
            self.update_model()

        inst_regrets = np.zeros((self.n_iter)) if max_val is not None else None

        for i in tqdm(range(self.n_iter)):
            # compute alpha once for this step
            x_next, temp_alpha = self.propose_location()

            # one-step-ahead prediction at chosen x (STRICTLY pre-data)
            #mu_pre, std_pre = self.predict_tempered(np.array(x_next).reshape(1, -1), temp_alpha)
            mu_pre, std_pre = self.predict_tempered(np.array(x_next).reshape(1, -1), 1.0)
            mu_pre = float(mu_pre.ravel()[0])
            var_pre = float(std_pre.ravel()[0]**2)

            # observe y
            noise = self.rng.normal(0, noise_std)
            y_next = float(self.func(x_next) + noise)

            # regret (using noise-free f)
            if max_val is not None:
                inst_regrets[i] = max_val - (y_next - noise)

            # update HW running sums (prequential)
            resid2 = (y_next - mu_pre)**2
            # EWMA noise estimate, use previous value in numerator to keep strictly one-step-ahead
            noise_prev = float(self._noise_hat2 if self._noise_hat2 is not None else self._noise_var)
            self._sum_num += var_pre + max(noise_prev, self._hw_eps)
            self._sum_den += var_pre + resid2
            # stability only
            tilde = max(0.0, resid2 - var_pre)
            self._noise_hat2 = (1.0 - self._hw_eta) * noise_prev + self._hw_eta * tilde

            # commit sample
            self.X[self.end_indx] = x_next
            self.Y[self.end_indx] = y_next

            # refit model and advance
            self.update_model()
            self.end_indx += 1
            self.current_indx += 1

            # store alpha used (optional; here we just keep self.alpha array as info)
            if self.alpha_type == "hw":
                self.alpha[self.current_indx-1] = temp_alpha

            if visualize:
                if self.dim > 1:
                    raise NotImplementedError("Only 1D visualize implemented")
                self.visualize(self.current_indx-1, self.end_indx)

            if verbose:
                print(f"Iter {i:2d}: x={x_next}, y={y_next:.4f}, best observed = {self.Y.max():.4f}")

        # final best
        if max_val is not None:
            best_regret_idx = np.argmin(inst_regrets)
        else: 
            best_regret_idx = None
        best_observed_y_idx = self.Y.argmax()
        result = {
                 "best_iteration_observed": best_observed_y_idx - self.n_input , 
                 "best_observed": self.Y.max(), 
                 "best_observed_x": self.X[best_observed_y_idx],
                 "best_iteration_regret": best_regret_idx,
                 "inst_regrets": inst_regrets
                 }
        return result


