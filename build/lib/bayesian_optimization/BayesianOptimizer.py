import numpy as np
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Matern,
    WhiteKernel,
)
from scipy.integrate import quad

class TemperedGPR(GaussianProcessRegressor):
    def __init__(self, *, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.current_alpha = alpha

    def log_marginal_likelihood(self, theta, eval_gradient=True):
        base = super().log_marginal_likelihood(theta, eval_gradient)
        return self.current_alpha * base

class BayesianOptimizer:
    def __init__(
        self,
        func,
        kernel,
        bounds: np.ndarray,
        x_init: np.ndarray,
        y_init: np.ndarray,
        g: int,
        alpha: float,
        xi: float = 0,
        n_iter=100,
        acq_samples=100,
        random_state=42,
    ):
        """
        @ Args
        func : true function
        bounds: d by 2 numpy array
        x_init: n by 2 numpy array
        y_init: n  by 1 dimensional array
        n_iter : total iteration
        acq_samples  : number of random candidates to sample per iteration
        random_state : int
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = self.bounds.shape[0]
        self.alpha, self.g = alpha, g
        self.n_iter = n_iter
        self.end_indx = n_iter
        self.acq_samples = acq_samples
        self.rng = np.random.RandomState(random_state)
        self.n_input = x_init.shape[0]
        self.X = np.zeros((n_iter + n_input, self.dim))
        self.X[:n_input] = x_init
        self.Y = np.zeros((n_iter + n_input, 1))
        self.Y[:n_input] = y_init.reshape(-1, 1)
        self.xi = xi

        # kern_signal = C(1.0, (1e-3, 1e3)) * Matern(
        #     length_scale=np.ones(self.dim),
        #     length_scale_bounds=(1e-2, 1e2),
        #     nu=2.5,
        # )
        # kern_noise = WhiteKernel(
        #     noise_level=1e-6,
        #     noise_level_bounds=(1e-10, 1e1),
        # )
        self.gp = TemperedGPR(
            alpha = alpha,
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.rng,
        )
        self._signal_kernel = None
        self._noise_var = None
    
    def update_model(self):
        """Refit the GP to all observed data."""
        self.gp.current_alpha = self.alpha
        X_train = self.X[: self.end_indx]
        y_train = self.Y[: self.end_indx].ravel()
        self.gp.fit(X_train, y_train)
        summed = self.gp.kernel_
        self._signal_kernel = summed.k1        
        self._noise_var = summed.k2.noise_level  

    def predict_tempered(self, X):
        """
        Make prediction under alpha posterior
        """
        Xcand = np.atleast_2d(X)
        X_train = self.X_sample[:self.end_indx]
        y_train = self.Y_sample[:self.end_indx].ravel()

        K_tt = self._signal_kernel(X_train, X_train)
        K_ts = self._signal_kernel(X_train, X)
        K_ss = self._signal_kernel(X, X)

        noise_mat = (self._noise_var / self.alpha) * np.eye(len(X_train))
        L, lower = cho_factor(K_tt + noise_mat, lower=True)
        alpha_vec = cho_solve((L, lower), y_train)
        mu = K_ts.T.dot(alpha_vec)

        K_inv_Kts = cho_solve((L, lower), K_ts)
        cov = K_ss - K_ts.T.dot(K_inv_Kts)
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))

        return mu, std

    def expected_improvement(self, x):
        """Standard Expected Improvement with Alpha Posterior Temperaing"""
        mu, sigma = self.predict_tempered(x)
        mu_sample_opt = np.max(self.Y[:self.end_indx])
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei.ravel()[0]  
    
   def general_expected_improvement(self, x):
        """General Expected Improvement with Alpha Posterior Temperaing"""
        mu, sigma = self.predict_tempered(x)
        mu_sample_opt = np.max(self.Y[:self.end_indx])
        v= (mu_sample_opt - mu)/sigma 
        integrand = lambda u: (u-v)**self.g * norm.pdf(u)
        result, _ = quad(integrand, v, np.inf, limit=100)
        return sigma**self.g * result

    def propose_location(self):
        """Sample random candidates and pick the one with largest EI."""
        best_x = None
        best_val = np.inf
        X_cand = self.rng.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(self.acq_samples, self.dim)
        )
        for x in X_cand:
            if self.g == 1:
                acq = lambda x: -self.expected_improvement(x)
            else:       
                acq = lambda x: -self.expected_improvement(x)
            res = minimize(acq, x0=x, bounds=self.bounds.tolist(),
                           method="L-BFGS-B")    
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        return best_x

    def simulate_optimization(self, noise_std = 0.01, verbose=True):
        self.update_model()
        for i in tqdm(range(self.n_input, self.n_input+self.n_iter)):
            x_next = self.propose_location()
            y_next = self.func(x_next) + self.rng.normal(0, self.noise_std)
            self.X[i] = x_next
            self.Y[i] = y_next
            self.update_model()
            self.end_indx += 1
            if verbose:
                print(f"Iter {i:2d}: x = {x_next}, y = {y_next:.4f}, "
                      f"best = {self.Y_sample.max():.4f}")

        # final best
        best_idx = np.argmax(self.Y_sample)
        return self.X_sample[best_idx], float(self.Y_sample[best_idx])

