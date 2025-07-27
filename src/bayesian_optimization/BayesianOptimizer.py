import numpy as np
from tqdm import tqdm
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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class TemperedGPR(GaussianProcessRegressor):
    def __init__(self, *, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.current_alpha = alpha

    def log_marginal_likelihood(self, theta, eval_gradient=True, clone_kernel=True,  **kwargs):
        res = super().log_marginal_likelihood(
            theta,
            eval_gradient=eval_gradient,
            clone_kernel=clone_kernel,
            **kwargs
        )
        lml, grad = res
        return self.current_alpha * lml, self.current_alpha * grad

class BayesianOptimizer:
    def __init__(
        self,
        func,
        kernel,
        bounds: np.ndarray,
        g: np.ndarray,
        alpha: np.ndarray,
        x_init: np.ndarray,
        y_init: np.ndarray,
        xi: float = 0,
        n_iter=100,
        acq_samples=100,
        random_state=42,
    ):
        """
        @ Args
        func : true function
        bounds: d by 2 numpy array
        x_init: n by d numpy array
        y_init: n  by 1 dimensional array
        n_iter : total iteration
        acq_samples  : number of random candidates to sample per iteration
        random_state : int
        """
        self.func = func
        self.bounds = bounds
        self.dim = self.bounds.shape[0]
        self.alpha, self.g = alpha, g
        self.n_iter = n_iter
        self.acq_samples = acq_samples
        self.rng = np.random.RandomState(random_state)
        self.n_input = x_init.shape[0]
        self.end_indx = self.n_input # end index for training
        self.current_indx = 0 # current iteartion
        self.X = np.zeros((n_iter + self.n_input, self.dim))
        self.Y = np.zeros((n_iter + self.n_input, 1))
        self.X[:self.n_input] = x_init
        self.Y[:self.n_input] = y_init.reshape(-1, 1)
        self.xi = xi

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
        self.gp.current_alpha = self.alpha[self.current_indx]
        X_train = self.X[: self.end_indx]
        y_train = self.Y[: self.end_indx].ravel()
        self.gp.fit(X_train, y_train)
        summed = self.gp.kernel_
        self._signal_kernel = summed.k1        
        self._noise_var = summed.k2.noise_level  
    
    def _ensure_2d(self, x):
        # x might be scalar, 1‑D, or already 2‑D
        arr = np.asarray(x)
        return arr.reshape(-1, self.dim)   # will give shape (n_points, dim)

    def predict_tempered(self, X, cur_alpha):
        """
        Make prediction under alpha posterior
        """
        X = np.asarray(X).reshape(-1, self.dim)
        X_train = self.X[:self.end_indx]
        y_train = self.Y[:self.end_indx].ravel()

        K_tt = self._signal_kernel(X_train, X_train)
        K_ts = self._signal_kernel(X_train, X)
        K_ss = self._signal_kernel(X, X)

        noise_mat = (self._noise_var / cur_alpha) * np.eye(len(X_train))
        L, lower = cho_factor(K_tt + noise_mat, lower=True)
        alpha_vec = cho_solve((L, lower), y_train)
        mu = K_ts.T.dot(alpha_vec)

        K_inv_Kts = cho_solve((L, lower), K_ts)
        cov = K_ss - K_ts.T.dot(K_inv_Kts)
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))

        return mu, std
    
    def probability_improvement(self, x, cur_alpha):
        """Standard Probablity Improvement with Alpha Posterior Temperaing"""
        #x = np.atleast_2d(x)
        x = self._ensure_2d(x)      
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_train, _  = self.predict_tempered(self.X[:self.end_indx], cur_alpha)
        mu_sample_opt  = np.max(mu_train)    
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - self.xi
            Z = imp / sigma
            pi = norm.cdf(Z) 
        if mu.shape[0] == 1:
            return pi.ravel()[0]  
        else:
            return pi.ravel()

    def expected_improvement(self, x, cur_alpha):
        """Standard Expected Improvement with Alpha Posterior Temperaing"""
        #x = np.atleast_2d(x)
        x = self._ensure_2d(x)      
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_train, _  = self.predict_tempered(self.X[:self.end_indx], cur_alpha)
        mu_sample_opt  = np.max(mu_train)    
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        if mu.shape[0] == 1:
            return ei.ravel()[0]  
        else:
            return ei.ravel()
    
    def power_2_expected_improvement(self, x, cur_alpha):
        """generalized expected improvement when g=2"""
        x = self._ensure_2d(x)      
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_train, _  = self.predict_tempered(self.X[:self.end_indx], cur_alpha)
        mu_sample_opt  = np.max(mu_train)    
        with np.errstate(divide='warn'):   
            imp = mu - mu_sample_opt - self.xi
            Z = imp / sigma
            result = sigma**2 *(norm.cdf(Z)*(1+Z**2)+Z*norm.pdf(Z))
        if mu.shape[0] == 1:
            return result.ravel()[0]  
        else:
            return result.ravel()
        
    
    def power_3_expected_improvement(self, x, cur_alpha):
        """generalized expected improvement when g=3"""
        x = self._ensure_2d(x)      
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_train, _  = self.predict_tempered(self.X[:self.end_indx], cur_alpha)
        mu_sample_opt  = np.max(mu_train)    
        with np.errstate(divide='warn'):   
            imp = mu - mu_sample_opt - self.xi
            Z = imp / sigma
            result = sigma**3 *(norm.cdf(Z)*(3*Z+Z**3)+(Z**2+2)*norm.pdf(Z))
        if mu.shape[0] == 1:
            return result.ravel()[0]  
        else:
            return result.ravel()
        
    
    def general_expected_improvement(self, x, cur_alpha):
        """General Expected Improvement with Alpha Posterior Temperaing
        
        Note currently only accept 1 input
        """
        x = self._ensure_2d(x)      
        mu, sigma = self.predict_tempered(x, cur_alpha)
        mu_train, _  = self.predict_tempered(self.X[:self.end_indx], cur_alpha)
        mu_sample_opt  = np.max(mu_train)   
        v= (mu_sample_opt - mu)/sigma 
        v= v.item()
        integrand = lambda u: (u-v)**self.g[self.current_indx] * norm.pdf(u)
        result, _ = quad(integrand, v, np.inf, limit=100)
        return(sigma**self.g[self.current_indx]).ravel()[0] * result

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
            if self.g[self.current_indx] == 1:
                acq = lambda x: -self.expected_improvement(x, self.alpha[self.current_indx])
            elif self.g[self.current_indx] ==0:
                acq = lambda x: -self.probability_improvement(x, self.alpha[self.current_indx])
            elif self.g[self.current_indx] == 2:
                acq = lambda x: -self.power_2_expected_improvement(x, self.alpha[self.current_indx])
            elif self.g[self.current_indx] == 3:
                acq = lambda x : -self.power_3_expected_improvement(x, self.alpha[self.current_indx])
            else:       
                acq = lambda x: -self.general_expected_improvement(x, self.alpha[self.current_indx])
            res = minimize(acq, x0=x, bounds=self.bounds.tolist(),
                           method="L-BFGS-B")    
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        return best_x
    

    def visualize(self, iteration_idx, data_end_idx):
        plt.figure(figsize=(8,4))
        x_star = np.linspace(self.bounds[0,0],self.bounds[0,1], 500)
        f_star = self.func(x_star)
        plt.plot(x_star, f_star, 'k-', label='true f(x)', alpha=0.5)
        data_x, data_y = self.X[:data_end_idx], self.Y[:data_end_idx]
        plt.scatter(data_x, data_y, c='C1', s=50, edgecolor='k')
        for i,(x,y) in enumerate(zip(data_x[self.n_input:],data_y[self.n_input:])):
            plt.text(x, y, str(i), color='black',
                    fontsize=9, ha='center', va='bottom')
        plt.xlabel('x'); plt.ylabel('f(x)')
        plt.title(f"iteartion{iteration_idx}: alpha={self.alpha[iteration_idx]}, g= {self.g[iteration_idx]}")
        mu, std = self.predict_tempered(x_star.reshape(-1, 1),self.alpha[iteration_idx])
        plt.plot(x_star, mu, 'b--', label='GP mean', alpha=0.5)
        # ±1σ band
        plt.fill_between(
            x_star.ravel(),
            (mu - std).ravel(),
            (mu + std).ravel(),
            color='blue',
            alpha=0.2,
            label=r'$\pm1\,\sigma$'
        )
        # plot acquization
        if self.g[iteration_idx] ==1:
            acq_y = self.expected_improvement(x_star, self.alpha[iteration_idx])
        print(acq_y)
        plt.plot(x_star, acq_y,  linestyle='--', color='C1', label='acquization')
        plt.legend()
        plt.show()

    def simulate_optimization(
            self, 
            noise_std = 0.01, 
            visualize = False,
            max_val=None, 
            verbose=True
        ):
        if self.n_input > 0:
            self.update_model()
        if max_val is not None:
            inst_regrets = np.zeros((self.n_iter))
        else:
            inst_regrets = None
        for i in tqdm(range(self.n_iter)):
            x_next = self.propose_location()
            noise = self.rng.normal(0, noise_std)
            y_next = self.func(x_next) + noise
            if max_val is not None:
                inst_regrets[i] = max_val - (y_next-noise)
            self.X[self.end_indx] = x_next
            self.Y[self.end_indx] = y_next
            if visualize:
                if self.dim > 1:
                    raise NotImplementedError("high-dimensioanl visualization has not been implemented")
                self.visualize(self.current_indx,self.end_indx)
            self.update_model()
            self.end_indx += 1
            self.current_indx += 1
            if verbose:
                print(f"Iter {i:2d}: x = {x_next}, y = {y_next}, "
                      f"best observed = {self.Y.max()}")
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

