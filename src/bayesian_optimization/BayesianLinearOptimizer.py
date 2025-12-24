import numpy as np
from typing import Callable, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------- RFF feature map (with ARD via input pre-scaling) --------------------
class RBFFeatureMap:
    """
    Random Fourier Features for the RBF kernel using scikit-learn's RBFSampler.
    """
    def __init__(
        self,
        input_dim: int,
        n_components: int = 512,
        lengthscale: Union[float, np.ndarray] = 1.0,
        include_bias: bool = True,
        random_state: Optional[int] = None,
    ):
        self.input_dim = int(input_dim)
        self.n_components = int(n_components)
        self.include_bias = bool(include_bias)
        self.random_state = random_state

        ell = np.asarray(lengthscale, dtype=float)
        if ell.ndim == 0:
            self.ell = np.full(self.input_dim, float(ell))
        else:
            if ell.size != self.input_dim:
                raise ValueError("lengthscale must be scalar or length = input_dim.")
            self.ell = ell

        # We always work in scaled inputs (x / ell) with gamma=1/2
        self.gamma = 0.5
        self._rff = RBFSampler(gamma=self.gamma, n_components=self.n_components, random_state=self.random_state)
        self._rff.fit(np.zeros((1, self.input_dim)))
        self.d_out = self.n_components + (1 if self.include_bias else 0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)  # (p,)
        if x.size != self.input_dim:
            raise ValueError(f"Expected x in R^{self.input_dim}, got {x.size}.")
        x_scaled = x / self.ell
        z = self._rff.transform(x_scaled.reshape(1, -1)).ravel()  # (D,)
        return np.concatenate([np.array([1.0]), z]) if self.include_bias else z


# -------------------- Tempered Bayesian Linear Regressor with explicit __init__ --------------------
# --- add to TemperedBLR ---
class TemperedBLR:
    def __init__(self, d, sigma2: float = 1e-2, lam: float = 1.0,
                 feature_map=None, current_alpha: float = 1.0,
                 sigma2_mode: str = "auto", max_evidence_iter: int = 25, tol: float = 1e-4):
        self.d = int(d)
        self.sigma2 = float(sigma2)
        self.lam = float(lam)
        self.feature_map = feature_map
        self.current_alpha = float(current_alpha)
        self.sigma2_mode = sigma2_mode  # "fixed" or "auto"
        self.max_evidence_iter = int(max_evidence_iter)
        self.tol = float(tol)
        self._Phi = None; self._y = None

    def fit(self, X, y):
        Phi = np.vstack([self.feature_map(xi) for xi in np.asarray(X)])
        if Phi.shape[1] != self.d:
            raise ValueError(f"Feature dim mismatch: expected d={self.d}, got {Phi.shape[1]}.")
        self._Phi = Phi
        self._y = np.asarray(y, float).ravel()
        if self.sigma2_mode == "auto" and self._Phi.shape[0] >= self.d:
            self._fit_evidence()  # update (lam, sigma2) like GP’s hyperparam fit

    def _fit_evidence(self):
        Phi, y = self._Phi, self._y
        n, d = Phi.shape
        # initialize (keep current), then iterate
        lam, sigma2 = max(self.lam, 1e-8), max(self.sigma2, 1e-12)
        for _ in range(self.max_evidence_iter):
            beta = 1.0 / max(sigma2, 1e-12)
            A = lam * np.eye(d) + beta * (Phi.T @ Phi)
            L, lower = cho_factor(A, lower=True)
            # Σ and m
            I = np.eye(d)
            Sigma = cho_solve((L, lower), I)
            m = beta * Sigma @ (Phi.T @ y)
            # gamma, residuals
            gamma = d - lam * np.trace(Sigma)
            gamma = float(np.clip(gamma, 0.0, d))
            resid = y - Phi @ m
            # updates
            lam_new = gamma / max(np.dot(m, m), 1e-12)
            beta_new = (n - gamma) / max(np.dot(resid, resid), 1e-12)
            lam_new = float(np.clip(lam_new, 1e-12, 1e12))
            sigma2_new = float(np.clip(1.0 / max(beta_new, 1e-12), 1e-12, 1e6))
            # check convergence
            if max(abs(lam_new - lam) / (lam + 1e-12),
                   abs(sigma2_new - sigma2) / (sigma2 + 1e-12)) < self.tol:
                lam, sigma2 = lam_new, sigma2_new
                break
            lam, sigma2 = lam_new, sigma2_new
        self.lam, self.sigma2 = lam, sigma2  # store

    def posterior_given_alpha(self, alpha: float):
        if self._Phi is None:
            raise RuntimeError("Model not fit yet.")
        # tempered posterior = use effective noise sigma2/alpha
        eff_beta = alpha / max(self.sigma2, 1e-12)
        A = self.lam * np.eye(self.d) + eff_beta * (self._Phi.T @ self._Phi)
        b = eff_beta * (self._Phi.T @ self._y)
        L, lower = cho_factor(A, lower=True)
        m = cho_solve((L, lower), b)
        Sigma = cho_solve((L, lower), np.eye(self.d))
        return m, Sigma
    
    def predict_latent(self, X: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        if self._Phi is None:
            raise RuntimeError("Model not fit yet.")
        m, Sigma = self.posterior_given_alpha(alpha)
        Psi = np.vstack([self.feature_map(xi) for xi in np.asarray(X)])
        mu = Psi @ m
        var = np.einsum("nd,dd,nd->n", Psi, Sigma, Psi, optimize=True)
        var = np.maximum(var, 1e-16)
        return mu.reshape(-1, 1), np.sqrt(var).reshape(-1, 1)



# -------------------- BLR Bayesian Optimizer (mirrors your GP class API) --------------------
class BayesianLinearOptimizer:
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        g: np.ndarray,                                
        alpha: Union[np.ndarray, str],               
        x_init: np.ndarray,
        y_init: np.ndarray,
        n_features: int = 512, # fourier dimension
        lengthscale: Union[float, np.ndarray] = 0.5,   # 
        include_bias: bool = True,
        sigma2: float = 1e-2,
        lam: float = 1.0,
        md_alpha: float = None,
        xi: float = 0.0,
        n_iter: int = 100,
        acq_samples: int = 10,
        random_state: int = 42,
        hw_clip: Tuple[float, float] = (0.01, 1.0),
        hw_eta: float = 0.1,
    ):
        self.func = func
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[0]

        self.n_iter = int(n_iter)
        self.acq_samples = int(acq_samples)
        self.rng = np.random.RandomState(random_state)

        self.n_input = x_init.shape[0]
        self.end_indx = self.n_input
        self.current_indx = 0
        self.X = np.zeros((self.n_input + self.n_iter, self.dim))
        self.Y = np.zeros((self.n_input + self.n_iter, 1))
        self.X[: self.n_input] = x_init
        self.Y[: self.n_input] = y_init.reshape(-1, 1)

        # schedules
        self.g = np.asarray(g, dtype=float)
        assert self.g.size >= self.n_iter, "g schedule must have length >= n_iter."
        self.xi = float(xi)

        if isinstance(alpha, np.ndarray):
            self.alpha = alpha.astype(float)
            self.alpha_type = None
            self.alpha_lb, self.alpha_ub = None, None
        else:
            self.alpha_lb, self.alpha_ub = 1e-2, 1.0
            if alpha == "hw":
                self.alpha = np.ones(self.n_iter, dtype=float)
            elif "md" in alpha:
                if md_alpha is None:
                    raise ValueError("For 'md' you must pass md_alpha.")
                self.alpha = np.array([md_alpha/(md_alpha+i)
                                       for i in range(self.n_input, self.n_input+self.n_iter)], dtype=float)
            else:
                raise ValueError("Unrecognized alpha argument!")
            self.alpha_type = alpha

        # RFF feature map (practitioner-friendly)
        self._feat = RBFFeatureMap(
            input_dim=self.dim,
            n_components=int(n_features),
            lengthscale=lengthscale,
            include_bias=include_bias,
            random_state=random_state,
        )
        self.d = self._feat.d_out

        # BLR surrogate
        self.blr = TemperedBLR(d=self.d, sigma2=float(sigma2), lam=float(lam),
                               feature_map=self._feat.transform, current_alpha=float(self.alpha[0]))

        # HW prequential stats
        self._hw_eps = 1e-12
        self._hw_eta = float(hw_eta)
        self._hw_clip = tuple(hw_clip)
        self._sum_num = self._hw_eps
        self._sum_den = self._hw_eps
        self._noise_hat2 = None  # EWMA of noise var (not used in BLR posterior, but for HW α estimate)

    def _ensure_2d(self, x):
        arr = np.asarray(x, dtype=float)
        return arr.reshape(-1, self.dim)

    def _sample_candidates(self, n=None):
        if n is None:
            n = self.acq_samples
        U = self.rng.uniform(size=(n, self.dim))
        return self.bounds[:, 0] + U * (self.bounds[:, 1] - self.bounds[:, 0])

    def get_hw_alpha(self):
        if self.current_indx == 0 or self._sum_den <= 0:
            return 1.0
        ratio = max(self._sum_num / self._sum_den, self._hw_eps)
        alpha = float(np.clip(np.sqrt(ratio), self._hw_clip[0], self._hw_clip[1]))
        return alpha

    def update_model(self):
        X_train = self.X[: self.end_indx]
        y_train = self.Y[: self.end_indx].ravel()
        self.blr.current_alpha = float(self.alpha[self.current_indx])
        self.blr.fit(X_train, y_train)
        if self._noise_hat2 is None:
            self._noise_hat2 =  max(1e-6, np.var(y_train))  # seed EWMA with something reasonable

    def predict_tempered(self, X, cur_alpha):
        X = self._ensure_2d(X)
        mu, std = self.blr.predict_latent(X, cur_alpha)
        return mu, std

    def probability_improvement(self, x, cur_alpha):
        x = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_train, _ = self.predict_tempered(self.X[:self.end_indx], cur_alpha)
        mu_star = np.max(mu_train)
        imp = mu - mu_star - self.xi
        Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=(sigma > 0))
        val = norm.cdf(Z)
        return val.ravel()[0] if mu.shape[0] == 1 else val.ravel()

    def expected_improvement(self, x, cur_alpha):
        x = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_train, _ = self.predict_tempered(self.X[:self.end_indx], cur_alpha)
        mu_star = np.max(mu_train)
        imp = mu - mu_star - self.xi
        Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=(sigma > 0))
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = np.where(sigma <= 0.0, 0.0, ei)
        return ei.ravel()[0] if mu.shape[0] == 1 else ei.ravel()

    def power_2_expected_improvement(self, x, cur_alpha):
        x = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_train, _ = self.predict_tempered(self.X[:self.end_indx], cur_alpha)
        mu_star = np.max(mu_train)
        imp = mu - mu_star - self.xi
        Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=(sigma > 0))
        val = sigma**2 * (norm.cdf(Z) * (1 + Z**2) + Z * norm.pdf(Z))
        return val.ravel()[0] if mu.shape[0] == 1 else val.ravel()

    def power_3_expected_improvement(self, x, cur_alpha):
        x = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_train, _ = self.predict_tempered(self.X[:self.end_indx], cur_alpha)
        mu_star = np.max(mu_train)
        imp = mu - mu_star - self.xi
        Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=(sigma > 0))
        val = sigma**3 * (norm.cdf(Z) * (3*Z + Z**3) + (Z**2 + 2) * norm.pdf(Z))
        return val.ravel()[0] if mu.shape[0] == 1 else val.ravel()

    def general_expected_improvement(self, x, cur_alpha):
        """General g-EI via 1D quadrature (same pattern as your GP code)."""
        x = self._ensure_2d(x)
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_train, _ = self.predict_tempered(self.X[:self.end_indx], cur_alpha)
        mu_star = np.max(mu_train)
        v = ((mu_star - mu) / sigma).item()
        g_now = float(self.g[self.current_indx])
        integrand = lambda u: (u - v)**g_now * norm.pdf(u)
        val, _ = quad(integrand, v, np.inf, limit=100)
        return float((sigma**g_now).ravel()[0] * val)

    def propose_location(self, temp_alpha=None):
        if temp_alpha is None:
            temp_alpha = self.get_hw_alpha() if (self.alpha_type == "hw") else float(self.alpha[self.current_indx])
        X_cand = self._sample_candidates(self.acq_samples)
        if self.g[self.current_indx] == 1:
            acq = lambda z: -self.expected_improvement(z, temp_alpha)
        elif self.g[self.current_indx] == 0:
            acq = lambda z: -self.probability_improvement(z, temp_alpha)
        elif self.g[self.current_indx] == 2:
            acq = lambda z: -self.power_2_expected_improvement(z, temp_alpha)
        elif self.g[self.current_indx] == 3:
            acq = lambda z: -self.power_3_expected_improvement(z, temp_alpha)
        else:
            acq = lambda z: -self.general_expected_improvement(z, temp_alpha)

        best_x, best_val = None, np.inf
        for x0 in X_cand:
            res = minimize(acq, x0=x0, bounds=self.bounds.tolist(), method="L-BFGS-B")
            if res.fun < best_val:
                best_val, best_x = res.fun, res.x
        return best_x, float(temp_alpha)


    def simulate_optimization(self, noise_std=0.0, visualize=False, max_val=None, verbose=True):
        if self.n_input > 0:
            self.update_model()

        inst_regrets = np.zeros((self.n_iter)) if max_val is not None else None

        for i in tqdm(range(self.n_iter)):
            x_next, temp_alpha = self.propose_location()

            # strictly pre-data predictive at chosen α
            mu_pre, std_pre = self.predict_tempered(np.array(x_next).reshape(1, -1), temp_alpha)
            mu_pre = float(mu_pre.ravel()[0]); var_pre = float(std_pre.ravel()[0] ** 2)

            # observe noisy y
            noise = np.random.default_rng().normal(0, noise_std)
            y_next = float(self.func(x_next) + noise)

            # simple regret proxy
            if max_val is not None:
                inst_regrets[i] = max_val - (y_next - noise)

            # update HW stats
            resid2 = (y_next - mu_pre) ** 2
            noise_prev = float(self._noise_hat2 if self._noise_hat2 is not None else self.sigma2)
            self._sum_num += var_pre + max(noise_prev, self._hw_eps)
            self._sum_den += var_pre + resid2
            tilde = max(0.0, resid2 - var_pre)
            self._noise_hat2 = (1.0 - self._hw_eta) * noise_prev + self._hw_eta * tilde

            # commit
            self.X[self.end_indx] = x_next
            self.Y[self.end_indx] = y_next

            # refit and advance
            self.update_model()
            self.end_indx += 1
            self.current_indx += 1
            if getattr(self, "alpha_type", None) == "hw":
                self.alpha[self.current_indx - 1] = temp_alpha

            if visualize and self.dim == 1:
                self.visualize(self.current_indx - 1, self.end_indx)

            if verbose:
                print(f"Iter {i:02d}: x={x_next}, y={y_next:.4f}, best_y={self.Y.max():.4f}")

        best_obs_idx = self.Y.argmax()
        result = {
            "best_iteration_observed": best_obs_idx - self.n_input,
            "best_observed": self.Y.max(),
            "best_observed_x": self.X[best_obs_idx],
            "inst_regrets": inst_regrets,
        }
        return result
