import pandas as pd
import numpy as np
import pathlib
import yaml
import argparse
import typing
import pickle
import bayesian_optimization as bo
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
        "--model_output_dir",
        default=pathlib.Path.home() / "BO" / "models" / "X01"
    )
    arguments = vars(parser.parse_args())
    with open(arguments["config_filepath"]) as f:
        config = yaml.full_load(f) or {}
    arguments.update(config)
    if arguments["verbose"]:
        print(arguments)
    return arguments


# Test configurations (simplified for multi-dimensional testing)
ALPHA_CONFIGS = [1.0, "hw"]
G_CONFIGS = [0, 1, 2]
NOISE_LEVELS = [0.01]  # Reduced for faster testing
SEEDS = [i for i in range(5)]  # Reduced for faster testing
DIMENSIONS = [5, 10]


def sphere_nd(x, dim=None):
    """N-dimensional sphere function: global optimum at center with value 0"""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    # Rescale from [0,1] to [-5,5]
    u = 10 * x - 5
    value = np.sum(u**2, axis=1)
    return -value

def rosenbrock_nd(x, dim=None):
    """N-dimensional Rosenbrock function"""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    # Rescale from [0,1] to [-2,2] 
    u = 4 * x - 2
    
    if u.shape[1] == 1:
        return -np.ones(u.shape[0])  # Degenerate case
    
    value = np.zeros(u.shape[0])
    for i in range(u.shape[1] - 1):
        value += 100 * (u[:, i+1] - u[:, i]**2)**2 + (1 - u[:, i])**2
    return -value


def ackley_nd(x, dim=None):
    """N-dimensional Ackley function"""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    # Rescale from [0,1] to [-5,5]
    u = 10 * x - 5
    
    n = u.shape[1]
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(u**2, axis=1) / n))
    term2 = -np.exp(np.sum(np.cos(2*np.pi*u), axis=1) / n)
    value = term1 + term2 + 20 + np.e
    return -value

def rastrigin_nd(x, dim=None):
    """N-dimensional Rastrigin function"""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    # Rescale from [0,1] to [-5.12,5.12]
    u = 10.24 * x - 5.12
    
    A = 10
    n = u.shape[1]
    value = A * n + np.sum(u**2 - A * np.cos(2*np.pi*u), axis=1)
    return -value

def schwefel_nd(x, dim=None):
    """N-dimensional Schwefel function (highly multimodal)"""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    # Rescale from [0,1] to [-500,500]
    u = 1000 * x - 500
    
    n = u.shape[1]
    value = 418.9829 * n - np.sum(u * np.sin(np.sqrt(np.abs(u))), axis=1)
    return -value

def levy_nd(x, dim=None):
    """N-dimensional Levy function"""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    # Rescale from [0,1] to [-10,10]
    u = 20 * x - 10
    
    w = 1 + (u - 1) / 4
    
    term1 = np.sin(np.pi * w[:, 0])**2
    
    if u.shape[1] > 1:
        term2 = np.sum((w[:, :-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1)**2), axis=1)
        term3 = (w[:, -1] - 1)**2 * (1 + np.sin(2 * np.pi * w[:, -1])**2)
    else:
        term2 = 0
        term3 = 0
    
    value = term1 + term2 + term3
    return -value


def zakharov_nd(x, dim=None):
    """Zakharov (nD). Domain: [-5,10]^d. Global min 0 at u=0 ⇒ x* = 1/3."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1: x = x.reshape(1, -1)
    u = 15.0 * x - 5.0
    d = u.shape[1]
    i = np.arange(1, d + 1, dtype=float)
    s1 = np.sum(u**2, axis=1)
    s2 = np.sum(0.5 * i * u, axis=1)
    value = s1 + s2**2 + s2**4
    return -value  # max = 0 at x = 1/3 * 1

def griewank_nd(x, dim=None):
    """Griewank (nD). Domain: [-600,600]^d. Global min 0 at u=0 ⇒ x* = 0.5."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1: x = x.reshape(1, -1)
    u = 1200.0 * x - 600.0
    d = u.shape[1]
    i = np.sqrt(np.arange(1, d + 1, dtype=float))
    sum_term = np.sum(u**2, axis=1) / 4000.0
    prod_term = np.prod(np.cos(u / i), axis=1)
    value = 1.0 + sum_term - prod_term
    return -value  # max = 0 at x = 0.5 * 1

def alpine1_nd(x, dim=None):   
    """Alpine 1 (nD). Domain: [0,10]^d. Global min 0 at u=0 ⇒ x* = 0 (boundary, still in [0,1]^d)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1: x = x.reshape(1, -1)
    u = 10.0 * x  # [0,10]^d
    value = np.sum(np.abs(u * np.sin(u) + 0.1 * u), axis=1)
    return -value  # max = 0 at x = 0 * 1

def expanded_scaffer_f6_nd(x, dim=None):
    """Expanded Scaffer F6 (nD, cyclic pairs). Domain: [-100,100]^d. Global min 0 at u=0 ⇒ x* = 0.5."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1: x = x.reshape(1, -1)
    u = 200.0 * x - 100.0
    # pairwise with wrap-around
    v = np.roll(u, shift=-1, axis=1)
    r2 = u**2 + v**2
    term = 0.5 + (np.sin(np.sqrt(r2))**2 - 0.5) / (1.0 + 0.001 * r2)**2
    value = np.sum(term, axis=1)
    return -value  # max = 0 at x = 0.5 * 1



FUNCTIONS = {
    'sphere': {'func': sphere_nd, 'optimum': 0.0, 'description': 'Convex, unimodal'},
    'rosenbrock': {'func': rosenbrock_nd, 'optimum': 0.0, 'description': 'Non-convex, valley'},
    'ackley': {'func': ackley_nd, 'optimum': 0.0, 'description': 'Highly multimodal'},
    'rastrigin': {'func': rastrigin_nd, 'optimum': 0.0, 'description': 'Many local minima'},
    'schwefel': {'func': schwefel_nd, 'optimum': 0.0, 'description': 'Deceptive, multimodal'},
    'levy': {'func': levy_nd, 'optimum': 0.0, 'description': 'Multimodal, steep ridges'},
    'zakharov': {'func': zakharov_nd,'optimum': 0.0,'description': 'Bowl + coupling term; non-separable, unimodal-ish'},
    'griewank': {'func': griewank_nd, 'optimum': 0.0, 'description': 'Weakly multimodal; widespread shallow local minima'},
    'alpine1': {'func': alpine1_nd,'optimum': 0.0,'description': 'Non-convex, multimodal (sine); optimum on boundary'},
    'expanded_scaffer_f6': {'func': expanded_scaffer_f6_nd,'optimum': 0.0,'description': 'Highly multimodal; cyclic pairwise non-separability'}
}

def create_experiment_config(dimensions=[2, 5, 10]):
    """Create experiment configuration for different dimensions"""
    
    configs = {}
    
    for dim in dimensions:
        # Scale parameters with dimension
        n_init = max(2*dim, 5)
        n_iter = min(10*dim, 200) #max(20, 5 * dim)  # Scale iterations with dimension
        acq_samples = 3 * dim  # Scale acquisition samples
        
        # Create bounds for this dimension
        bounds = np.array([[0, 1] for _ in range(dim)])
        # Create kernel for this dimension
        # kernel = (C(1.0, (1e-4, 1e4)) * 
        #           Matern(length_scale=np.ones(dim), 
        #                 length_scale_bounds=(1e-2, 1e2), nu=2.5) + 
        #           WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1)))

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



def run_single_optimization(func_name, func_info, config, alpha_config, g_config, noise_std, seed):
    """Run a single optimization experiment for given dimension"""
    
    # Setup
    func = func_info['func']
    true_optimum = func_info['optimum']
    dim = config['dim']
    n_iter = config['n_iter']
    
    # Generate g and alpha vectors
    if g_config == "step_decrease":
        mid = n_iter//2
        remainder = n_iter - mid
        g_vec = np.concatenate([np.ones(mid), np.zeros(remainder)])
    else:
        g_vec = np.ones(n_iter) * g_config
        
    # Handle alpha configuration properly (fixes the "md" in alpha bug)
    if isinstance(alpha_config, (int, float)):
        alpha_vec = np.ones(n_iter) * alpha_config
    else:
        alpha_vec = alpha_config
    
    # Initial data
    rng = np.random.RandomState(seed)
    x_init = rng.uniform(0, 1, size=(config['n_init'], dim))
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
        noise_std=noise_std,
        max_val=true_optimum,
        verbose=False
    )
    
    # Extract best-so-far trajectory
    Y_observed = optimizer.Y[:optimizer.end_indx].flatten()
    best_so_far = np.array([np.max(Y_observed[:i+1]) for i in range(len(Y_observed))])
    
    return {
        'function': func_name,
        'dimension': dim,
        'alpha': str(alpha_config),
        'g': str(g_config),
        'noise_std': noise_std,
        'seed': seed,
        'best_so_far': best_so_far,
        'final_best': result['best_observed'],
        'regret': true_optimum - best_so_far,
        'final_regret': true_optimum - result['best_observed'],
        'n_evaluations': len(best_so_far)
    }


def analyze_results(results):
    """Analyze multi-dimensional results"""
    
    df_data = []
    for result in results:
        df_data.append({
            'function': result['function'],
            'dimension': result['dimension'],
            'alpha': result['alpha'],
            'g': result['g'],
            'noise_std': result['noise_std'],
            'seed': result['seed'],
            'final_best': result['final_best'],
            'final_regret': result['final_regret'],
        })
    
    df = pd.DataFrame(df_data)
    
    print("\n" + "="*80)
    print("MULTI-DIMENSIONAL BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    # Performance by dimension
    print(f"\nPerformance by dimension (mean final best value):")
    print("-" * 50)
    
    dim_performance = df.groupby('dimension')['final_best'].agg(['mean', 'std', 'count'])
    for dim, stats in dim_performance.iterrows():
        print(f"{dim:>3}D: {stats['mean']:8.4f} ± {stats['std']:6.4f} "
              f"({stats['count']} experiments)")
    
    # Best method per function per dimension
    print(f"\nBest configuration per function-dimension:")
    print("-" * 60)
    
    for func_name in FUNCTIONS.keys():
        print(f"\n{func_name.upper()}:")
        func_results = df[df['function'] == func_name]
        
        for dim in DIMENSIONS:
            dim_results = func_results[func_results['dimension'] == dim]
            if len(dim_results) > 0:
                best_config = dim_results.groupby(['alpha', 'g', 'noise_std'])['final_best'].mean().idxmax()
                best_value = dim_results.groupby(['alpha', 'g', 'noise_std'])['final_best'].mean().max()
                
                alpha, g, noise = best_config
                print(f"  {dim:2}D: α={alpha}, g={g}, σ={noise:.3f} → {best_value:8.4f}")
    
    # Difficulty ranking by dimension
    print(f"\nFunction difficulty ranking by dimension (harder = lower final best):")
    print("-" * 70)
    
    for dim in DIMENSIONS:
        print(f"\n{dim}D:")
        dim_results = df[df['dimension'] == dim]
        difficulty = dim_results.groupby('function')['final_best'].mean().sort_values()
        
        for rank, (func, score) in enumerate(difficulty.items(), 1):
            print(f"  {rank}. {func:>12}: {score:8.4f}")
    
    return df


def run_benchmark(model_output_dir: pathlib.Path):
    """Run complete multi-dimensional benchmark suite"""
    
    print("Running Multi-Dimensional Bayesian Optimization Benchmark")
    print("="*60)
    
    # Create configurations for different dimensions
    configs = create_experiment_config(DIMENSIONS)
    
    results = []
    total_experiments = (len(FUNCTIONS) * len(DIMENSIONS) * len(ALPHA_CONFIGS) * 
                        len(G_CONFIGS) * len(NOISE_LEVELS) * len(SEEDS))
    
    print(f"Dimensions tested: {DIMENSIONS}")
    print(f"Functions: {list(FUNCTIONS.keys())}")
    print(f"Total experiments: {total_experiments}")
    print()
    
    experiment_count = 0
    
    for dim in DIMENSIONS:
        config = configs[dim]
        print(f"\n--- DIMENSION {dim}D ---")
        
        for func_name, func_info in FUNCTIONS.items():
            print(f"Testing {func_name} ({func_info['description']})...")
            
            for alpha in ALPHA_CONFIGS:
                for g in G_CONFIGS:
                    for noise in NOISE_LEVELS:
                        for seed in SEEDS:
                            experiment_count += 1
                            print(f"  Exp {experiment_count}/{total_experiments}: "
                                  f"α={alpha}, g={g}, σ={noise:.3f}, seed={seed}", end="")
                            
                            try:
                                result = run_single_optimization(
                                    func_name, func_info, config, alpha, g, noise, seed
                                )
                                results.append(result)
                                print(" ✓")
                            except Exception as e:
                                print(f" ✗ ERROR: {str(e)[:50]}...")
                                continue
        
    # save result for each dimension  
            with open(model_output_dir.joinpath('s01_batch1_experiment_results.pkl'), 'wb') as f:
                pickle.dump(results, f)
    df = analyze_results(results)
    df.to_csv(model_output_dir.joinpath('s01_batch1_experiment_results.csv'))
    
    return results

if __name__ == "__main__":
    # data setup
    arguments = parse_args()
    print(arguments["model_output_dir"])
    results = run_benchmark(model_output_dir = arguments["model_output_dir"])
   



    
    

