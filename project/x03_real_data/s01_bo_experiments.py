import pandas as pd
import numpy as np
import pathlib
import yaml
import argparse
import typing
import pickle
import bayesian_optimization as bo
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import seaborn as sns
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Matern,
    WhiteKernel,
)
import pickle
import copy


ALPHA_CONFIGS = [1.0, "hw"]
G_CONFIGS = ["MES", 0, 0.5, 1, 1.5, 2]
SEEDS = [i for i in range(10)]  


def parse_args() -> typing.Dict[str, typing.Any]:
    parser = argparse.ArgumentParser("Fit BO on Function Classes")
    parser.add_argument(
        "--config_filepath",
        default=pathlib.Path.home() / "BO" / "config" / "config.yaml"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False
    )
    parser.add_argument(
        "--data_input_dir",
        default=pathlib.Path.home() / "BO" / "data" / "voltage"
    )
    parser.add_argument(
        "--model_output_dir",
        default=pathlib.Path.home() / "BO" / "models" / "X03"
    )
    arguments = vars(parser.parse_args())
    with open(arguments["config_filepath"]) as f:
        config = yaml.full_load(f) or {}
    arguments.update(config)
    if arguments["verbose"]:
        print(arguments)
    return arguments


def create_experiment_config(dim=2):
    """Create experiment configuration for different dimensions
    
    Note although it's 3-dimensional problem, the degrees of freedom is 2
    """
    configs = {}
    n_init = 5
    n_iter = 30 
    acq_samples = 10  
    bounds = np.array([[0, 1] for _ in range(dim)])
    kernel = (
        C(1.0, (1e-3, 1e3)) *
        Matern(length_scale=np.full(dim, 0.3),
            length_scale_bounds=(1e-3, 100), nu=2.5)
        +
        WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e1))
    )
    configs[dim] = {
        'dim': dim,
        'n_init': n_init,
        'n_iter': n_iter,
        'acq_samples': acq_samples,
        'bounds': bounds,
        'kernel': kernel
    }
    
    return configs



def run_single_optimization(func_info, config, alpha_config, g_config, seed):
    """Run a single optimization experiment for given dimension"""
    
    func = func_info['func']
    true_optimum = func_info['optimum']
    dim = config['dim'] # dimensio is 2
    n_iter = config['n_iter']
    
    # Generate g and alpha vectors
    if g_config == "step_decrease":
        mid = n_iter//2
        remainder = n_iter - mid
        g_vec = np.concatenate([np.ones(mid), np.zeros(remainder)])
    elif g_config == "MES":
        g_vec = np.array(["MES"]*n_iter)
    else:
        g_vec = np.ones(n_iter) * g_config
        
    # Handle alpha configuration properly (fixes the "md" in alpha bug)
    if isinstance(alpha_config, (int, float)):
        alpha_vec = np.ones(n_iter) * alpha_config
    else:
        alpha_vec = alpha_config
    
    # Initial data
    rng = np.random.RandomState(seed)
    #x_init = rng.uniform(0, 1, size=(config['n_init'], dim))
    X_init = rng.dirichlet(alpha=np.ones(dim+1), size=config["n_init"]) 
    # Take the first two components
    x_init = X_init[:, :dim]   
    y_init = func(x_init).flatten()
    # Create optimizer
    optimizer = bo.BayesianOptimizer(
        func=func,
        kernel= copy.deepcopy(config['kernel']),
        bounds=config['bounds'],
        g=g_vec,
        alpha=alpha_vec,
        x_init=x_init,
        y_init=y_init,
        n_iter=n_iter,
        acq_samples=config['acq_samples'],
        random_state=seed
    )
    
    # Run optimization
    result = optimizer.simulate_optimization(
        noise_std=0.01,
        max_val=true_optimum,
        verbose=False
    )
    
    # Extract best-so-far trajectory
    Y_observed = optimizer.Y[:optimizer.end_indx].flatten()
    best_so_far = np.array([np.max(Y_observed[:i+1]) for i in range(len(Y_observed))])
    
    return {
        'dimension': dim,
        'alpha': str(alpha_config),
        'g': str(g_config),
        'noise_std': 0.01,
        'seed': seed,
        'best_so_far': best_so_far,
        'final_best': result['best_observed'],
        'regret': true_optimum - best_so_far,
        'final_regret': true_optimum - result['best_observed'],
        'n_evaluations': len(best_so_far)
    }


def run_benchmark(experiment_config, model_output_dir: pathlib.Path, func_info, dim=2):
    """Run complete multi-dimensional benchmark suite"""
    results = []
    config = experiment_config[dim]
    experiment_count = 0
    for alpha in ALPHA_CONFIGS:
        for g in G_CONFIGS:
            for seed in SEEDS:
                experiment_count += 1
                print(f"----------------Experiment {experiment_count}---------------")
                print(alpha, g, seed)
                try:
                    result = run_single_optimization(
                       func_info, config, alpha, g, seed
                    )
                    results.append(result)
                    print(" ✓")
                except Exception as e:
                    print(f" ✗ ERROR: {str(e)[:50]}...")
                    continue

    with open(model_output_dir.joinpath('s01_voltage_experiment_results_30.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    return results


def make_gp_mean_fn_simplex2D(model, penalty=-1e12, tol=1e-12):
    """
    mean_fun2D([Fe, Ga]) with Pd = 1 - Fe - Ga.
    Returns a scalar for shape (2,), or a 1D array for (n,2).
    Infeasible points (Fe<0, Ga<0, Pd<0) return `penalty`.
    """
    def mean_fun2D(x2):
        X2 = np.asarray(x2, dtype=float)
        is_1d = (X2.ndim == 1)
        X2 = X2[None, :] if is_1d else X2

        Fe = X2[:, 0]
        Ga = X2[:, 1]
        Pd = 1.0 - Fe - Ga
        X3 = np.stack([Fe, Ga, Pd], axis=1)

        valid = (X3 >= -tol).all(axis=1)
        out = np.empty(X2.shape[0], dtype=float)
        out[~valid] = penalty
        if valid.any():
            out[valid] = model.predict(X3[valid], return_std=False).ravel()

        return float(out[0]) if is_1d else out
    return mean_fun2D


def simplex_grid_2d(grid_n: int):
    """
    Triangular lattice on the simplex boundary with step 1/grid_n:
      G2 = {(Fe,Ga): Fe = i/n, Ga = j/n, i,j >= 0, i+j <= n}
    Returns: (N,2) array with N = (n+1)(n+2)/2.
    """
    i, j = np.indices((grid_n + 1, grid_n + 1))
    mask = (i + j) <= grid_n
    Fe = (i[mask] / grid_n).astype(float)
    Ga = (j[mask] / grid_n).astype(float)
    return np.column_stack([Fe, Ga])

def optimize_gp_mean(mean_fun, grid_n: int = 300, return_3d: bool = True):
    """
    Exhaustive grid search over the Fe–Ga simplex for the GP posterior mean.
    mean_fun: callable that accepts (2,) or (N,2) -> scalar / (N,)
              (e.g., from make_gp_mean_fn_simplex2D)
    grid_n:   lattice resolution; larger => finer search (O(grid_n^2))
    return_3d: if True, also return (Fe,Ga,Pd) with Pd = 1 - Fe - Ga
    """
    G2 = simplex_grid_2d(grid_n)           # (N,2)
    vals = mean_fun(G2).ravel()            # (N,)
    idx = int(np.argmax(vals))
    x2_best = G2[idx]
    f_best  = float(vals[idx])
    if return_3d:
        Fe, Ga = x2_best
        Pd = 1.0 - Fe - Ga
        x3_best = np.array([Fe, Ga, Pd], dtype=float)
        return x3_best, f_best
    else:
        return x2_best, f_best

def get_func_info(voltage_data:pd.DataFrame):
    """Fit GP model on Voltage Data, Use Posterior Mean as Ground Trueth and then Find the Optimal Value"""
    X = voltage_data[['Fe', 'Ga', 'Pd']].to_numpy()
    y = voltage_data['saturated_voltage'].to_numpy()
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.2, 0.2, 0.2],
                                        length_scale_bounds=(1e-3, 1e3),
                                        nu=2.5) 
          #  + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-8,
                                normalize_y=True,
                                n_restarts_optimizer=8,
                                random_state=0)
    gpr.fit(X, y)
    # Use posterior mean as the true function
    gp_mean = make_gp_mean_fn_simplex2D(gpr)
    # find optimal of the function
    opt_loc, optimal_val = optimize_gp_mean(gp_mean)
    return {"func":gp_mean, "optimum":optimal_val, "opt_loc": opt_loc}


def read_and_process_data(data_input_dir:pathlib.Path):
        """Process material and Voltage data"""
        material_df = pd.read_csv(
            data_input_dir.joinpath("FeGaPd_CMP.txt"),
            sep=r"\s+",         
            engine="python"
        ) 
        mag_df = pd.read_csv(
            data_input_dir.joinpath("FeGaPd_Mag.txt"),
            header=None,
            names=["voltage", "saturated_voltage"]
        ) 

        # 3) Align lengths (pads the shorter one with NaNs so concat never fails)
        rows = max(len(material_df), len(mag_df))
        material_df = material_df.reset_index(drop=True).reindex(range(rows))
        mag_df = mag_df.reset_index(drop=True).reindex(range(rows))

        # 4) Horizontal merge (column-wise concat)
        merged = pd.concat([material_df, mag_df], axis=1)
        return merged
    

if __name__ == "__main__":
    # data setup
    arguments = parse_args()
    experiment_config = create_experiment_config()
    data = read_and_process_data(arguments["data_input_dir"])
    func_info = get_func_info(data)
    optimum = {"max": func_info["optimum"], "argmax": func_info["opt_loc"]}
    results = run_benchmark(experiment_config=experiment_config, model_output_dir = arguments["model_output_dir"], func_info = func_info)
    results_all = {"experiments": results, "opt": optimum}
    with open(arguments["model_output_dir"].joinpath('s01_voltage_experiment_results_30.pkl'), 'wb') as f:
         pickle.dump(results_all, f)



    
    

