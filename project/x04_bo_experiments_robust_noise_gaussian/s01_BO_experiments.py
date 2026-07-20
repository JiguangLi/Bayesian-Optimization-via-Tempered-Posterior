import argparse
import copy
import pathlib
import pickle
import typing
import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc
import bayesian_optimization as bo
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel


def parse_args() -> typing.Dict[str, typing.Any]:
    parser = argparse.ArgumentParser("Fit BO on Function Classes")
    parser.add_argument(
        "--config_filepath",
        default=pathlib.Path.home() / "BO" / "config" / "config.yaml",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--model_output_dir",
        default=pathlib.Path.home() / "BO" / "models" / "X04",
    )
    parser.add_argument(
        "--noise_model",
        choices=["gaussian", "student_t", "gaussian_mixture", "heteroscedastic"],
        default="gaussian",
        help="Override NOISE_MODELS with a single noise model.",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=None,
        help="Override NOISE_LEVELS with a single baseline standard deviation.",
    )
    parser.add_argument("--student_t_df", type=float, default=4.0)
    parser.add_argument("--mixture_contamination", type=float, default=0.10) # pi \epsilon \sim (1-\pi) N(0, \sigma_in^2) + \pi N(0, (c \sigma_in)^2)
    parser.add_argument("--mixture_scale", type=float, default=10.0) # c
    parser.add_argument("--hetero_peak_multiplier", type=float, default=8.0) # rho
    parser.add_argument("--hetero_lengthscale_frac", type=float, default=0.02) # kappa
    parser.add_argument("--hetero_mc_samples", type=int, default=20000)

    arguments = vars(parser.parse_args())
    with open(arguments["config_filepath"]) as f:
        config = yaml.full_load(f) or {}
    arguments.update(config)
    if arguments["verbose"]:
        print(arguments)
    return arguments


ALPHA_CONFIGS = [1.0, "hw"]
G_CONFIGS = [0, 1, 2]
NOISE_LEVELS = [2]
NOISE_MODELS = ["gaussian"]
SEEDS = [i for i in range(5)]
DIMENSIONS = [5, 10]

# Noise-model defaults. All non-Gaussian settings preserve the domain-average or marginal
# standard deviation at ``base_std`` so the comparison to the Gaussian baseline is apples-to-apples.
STUDENT_T_DF = 4.0
GAUSSIAN_MIXTURE_CONTAMINATION = 0.10
GAUSSIAN_MIXTURE_SCALE = 10.0
HETERO_PEAK_MULTIPLIER = 8.0
HETERO_LENGTHSCALE_FRAC = 0.02
HETERO_NORMALIZER_MC_SAMPLES = 20000

# Optional exact optimizers for the heteroscedastic variance bump. Leave empty and the code will
# fall back to a deterministic Sobol pilot search on the noiseless benchmark.
KNOWN_OPTIMUM_X = {}

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


# -----------------------------------------------------------------------------
# Noise models
# -----------------------------------------------------------------------------
# Gaussian:
#   epsilon ~ N(0, sigma^2).
#
# Student-t heavy-tail:
#   epsilon = sigma * sqrt((nu - 2) / nu) * T_nu,  with nu = 4.
#   Because Var(T_nu) = nu / (nu - 2), this scaling preserves Var(epsilon) = sigma^2.
#
# Contaminated Gaussian mixture:
#   epsilon ~ (1 - pi) N(0, sigma_in^2) + pi N(0, (c * sigma_in)^2),
#   pi = 0.10, c = 10, sigma_in = sigma / sqrt((1 - pi) + pi * c^2).
#   This preserves the marginal variance sigma^2 while injecting occasional high-variance shocks.
#
# Heteroscedastic Gaussian:
#   epsilon(x) ~ N(0, sigma^2 * lambda(x) / Z),
#   lambda(x) = 1 + rho * exp(-||x - x_ref||^2 / (kappa * d)),
#   rho = 8, kappa = 0.02, and Z = E_U[lambda(U)] for U ~ Unif([0,1]^d).
#   We estimate Z once by deterministic Monte Carlo so the domain-average variance remains sigma^2.
#   The bump is centered at a benchmark-specific reference optimizer x_ref when available and
#   otherwise at a deterministic Sobol-search estimate of the noiseless optimum.
# -----------------------------------------------------------------------------

_HETERO_REFERENCE_CACHE = {}
_HETERO_NORMALIZER_CACHE = {}


def get_selected_noise_models(arguments: typing.Dict[str, typing.Any]):
    if arguments.get("noise_model") is not None:
        return [str(arguments["noise_model"])]
    return list(NOISE_MODELS)


def get_selected_noise_levels(arguments: typing.Dict[str, typing.Any]):
    if arguments.get("noise_std") is not None:
        return [float(arguments["noise_std"])]
    return list(NOISE_LEVELS)


def get_noise_hyperparameters(arguments: typing.Dict[str, typing.Any]):
    return {
        "student_t_df": float(arguments.get("student_t_df", STUDENT_T_DF)),
        "mixture_contamination": float(arguments.get("mixture_contamination", GAUSSIAN_MIXTURE_CONTAMINATION)),
        "mixture_scale": float(arguments.get("mixture_scale", GAUSSIAN_MIXTURE_SCALE)),
        "hetero_peak_multiplier": float(arguments.get("hetero_peak_multiplier", HETERO_PEAK_MULTIPLIER)),
        "hetero_lengthscale_frac": float(arguments.get("hetero_lengthscale_frac", HETERO_LENGTHSCALE_FRAC)),
        "hetero_mc_samples": int(arguments.get("hetero_mc_samples", HETERO_NORMALIZER_MC_SAMPLES)),
    }


def noise_model_summary(noise_model: str, base_std: float, noise_hyperparams: typing.Dict[str, typing.Any]):
    if noise_model == "gaussian":
        return f"Gaussian: epsilon ~ N(0, {base_std:.4f}^2)"
    if noise_model == "student_t":
        nu = noise_hyperparams["student_t_df"]
        return (
            "Student-t: epsilon = sigma * sqrt((nu-2)/nu) * T_nu "
            f"with sigma={base_std:.4f}, nu={nu:.1f}"
        )
    if noise_model == "gaussian_mixture":
        pi = noise_hyperparams["mixture_contamination"]
        scale = noise_hyperparams["mixture_scale"]
        return (
            "Contaminated Gaussian: epsilon ~ (1-pi)N(0,sigma_in^2)+piN(0,(c sigma_in)^2) "
            f"with sigma={base_std:.4f}, pi={pi:.2f}, c={scale:.1f}"
        )
    if noise_model == "heteroscedastic":
        rho = noise_hyperparams["hetero_peak_multiplier"]
        kappa = noise_hyperparams["hetero_lengthscale_frac"]
        return (
            "Heteroscedastic Gaussian: epsilon(x) ~ N(0, sigma^2 lambda(x)/Z), "
            f"lambda(x)=1+{rho:.2f} exp(-||x-x_ref||^2/({kappa:.4f} d))"
        )
    raise ValueError(f"Unknown noise model: {noise_model}")


def _noise_slug(noise_models: typing.Sequence[str]):
    if len(noise_models) == 1:
        return noise_models[0]
    return "multi_noise"


def _serialize_noise_spec(noise_spec: typing.Dict[str, typing.Any]):
    serialized = {}
    for key, value in noise_spec.items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            serialized[key] = value.item()
        else:
            serialized[key] = value
    return serialized


def _ensure_2d(x, dim=None):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        if dim is None:
            raise ValueError("dim must be supplied when reshaping a 1D input.")
        x = x.reshape(1, dim)
    return x


def _estimate_reference_optimum(func_name, func, dim, n_candidates=None, seed=2026):
    key = (func_name, int(dim))
    if key in _HETERO_REFERENCE_CACHE:
        return _HETERO_REFERENCE_CACHE[key].copy()

    n_candidates = int(max(4096, 1024 * dim) if n_candidates is None else n_candidates)
    m = int(2 ** np.ceil(np.log2(n_candidates)))
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed + 17 * dim + len(func_name))
    X = sampler.random_base2(m=int(np.log2(m)))
    y = np.asarray(func(X), dtype=float).reshape(-1)
    x_ref = X[int(np.argmax(y))].astype(float)
    _HETERO_REFERENCE_CACHE[key] = x_ref.copy()
    return x_ref


def _get_reference_optimum_x(func_name, func_info, dim):
    if func_info.get("optimum_x") is not None:
        x_ref = np.asarray(func_info["optimum_x"], dtype=float).reshape(-1)
        if x_ref.size == 1:
            x_ref = np.repeat(x_ref.item(), dim)
        if x_ref.size != dim:
            raise ValueError(
                f"optimum_x for {func_name} has length {x_ref.size}, expected {dim}."
            )
        return x_ref, "registry"

    if func_name in KNOWN_OPTIMUM_X:
        x_ref = np.asarray(KNOWN_OPTIMUM_X[func_name](dim), dtype=float).reshape(-1)
        if x_ref.size != dim:
            raise ValueError(
                f"KNOWN_OPTIMUM_X for {func_name} has length {x_ref.size}, expected {dim}."
            )
        return x_ref, "known_optimum_map"

    return _estimate_reference_optimum(func_name, func_info["func"], dim), "sobol_pilot"


def _hetero_multiplier(X, center, peak_multiplier, lengthscale_frac):
    center = np.asarray(center, dtype=float).reshape(-1)
    X = _ensure_2d(X, dim=center.size)
    sqdist = np.sum((X - center.reshape(1, -1)) ** 2, axis=1)
    denom = max(float(lengthscale_frac) * X.shape[1], 1e-12)
    return 1.0 + float(peak_multiplier) * np.exp(-sqdist / denom)


def _estimate_hetero_normalizer(center, dim, peak_multiplier, lengthscale_frac, mc_samples=20000, seed=314159):
    key = (
        tuple(np.round(np.asarray(center, dtype=float).reshape(-1), 8)),
        int(dim),
        float(peak_multiplier),
        float(lengthscale_frac),
        int(mc_samples),
    )
    if key in _HETERO_NORMALIZER_CACHE:
        return _HETERO_NORMALIZER_CACHE[key]

    m = int(2 ** np.ceil(np.log2(max(1024, int(mc_samples)))))
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed + 13 * dim)
    U = sampler.random_base2(m=int(np.log2(m)))
    normalizer = float(np.mean(_hetero_multiplier(U, center, peak_multiplier, lengthscale_frac)))
    _HETERO_NORMALIZER_CACHE[key] = normalizer
    return normalizer


def build_noise_spec(
    func_name: str,
    func_info: typing.Dict[str, typing.Any],
    dim: int,
    noise_model: str,
    base_std: float,
    noise_hyperparams: typing.Dict[str, typing.Any],
):
    base_std = float(base_std)
    noise_model = str(noise_model)

    if noise_model == "gaussian":
        return {
            "name": "gaussian",
            "base_std": base_std,
            "formula": "epsilon ~ N(0, sigma^2)",
        }

    if noise_model == "student_t":
        df = float(noise_hyperparams["student_t_df"])
        if df <= 2:
            raise ValueError("student_t_df must be > 2 so the variance is finite.")
        scale = base_std * np.sqrt((df - 2.0) / df)
        return {
            "name": "student_t",
            "base_std": base_std,
            "df": df,
            "scale": float(scale),
            "formula": "epsilon = sigma * sqrt((nu-2)/nu) * T_nu",
        }

    if noise_model == "gaussian_mixture":
        pi = float(noise_hyperparams["mixture_contamination"])
        c = float(noise_hyperparams["mixture_scale"])
        if not (0.0 < pi < 1.0):
            raise ValueError("mixture_contamination must lie in (0,1).")
        if c <= 1.0:
            raise ValueError("mixture_scale must exceed 1 so the contaminated component is noisier.")
        inlier_std = base_std / np.sqrt((1.0 - pi) + pi * (c ** 2))
        return {
            "name": "gaussian_mixture",
            "base_std": base_std,
            "contamination_prob": pi,
            "contamination_scale": c,
            "inlier_std": float(inlier_std),
            "outlier_std": float(c * inlier_std),
            "formula": "epsilon ~ (1-pi)N(0,sigma_in^2) + pi N(0,(c sigma_in)^2)",
        }

    if noise_model == "heteroscedastic":
        peak_multiplier = float(noise_hyperparams["hetero_peak_multiplier"])
        lengthscale_frac = float(noise_hyperparams["hetero_lengthscale_frac"])
        mc_samples = int(noise_hyperparams["hetero_mc_samples"])
        center, center_source = _get_reference_optimum_x(func_name, func_info, dim)
        normalizer = _estimate_hetero_normalizer(
            center=center,
            dim=dim,
            peak_multiplier=peak_multiplier,
            lengthscale_frac=lengthscale_frac,
            mc_samples=mc_samples,
        )
        return {
            "name": "heteroscedastic",
            "base_std": base_std,
            "center": center.copy(),
            "center_source": center_source,
            "peak_multiplier": peak_multiplier,
            "lengthscale_frac": lengthscale_frac,
            "normalizer": float(normalizer),
            "formula": "epsilon(x) ~ N(0, sigma^2 lambda(x)/Z), lambda(x)=1+rho exp(-||x-x_ref||^2/(kappa d))",
        }

    raise ValueError(f"Unknown noise model: {noise_model}")


def sample_noise(X, noise_spec: typing.Dict[str, typing.Any], rng: np.random.RandomState):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n = X.shape[0]
    name = noise_spec["name"]
    base_std = float(noise_spec["base_std"])

    if name == "gaussian":
        return rng.normal(0.0, base_std, size=n)

    if name == "student_t":
        return float(noise_spec["scale"]) * rng.standard_t(df=float(noise_spec["df"]), size=n)

    if name == "gaussian_mixture":
        mask = rng.uniform(size=n) < float(noise_spec["contamination_prob"])
        stds = np.where(mask, float(noise_spec["outlier_std"]), float(noise_spec["inlier_std"]))
        return rng.normal(0.0, stds, size=n)

    if name == "heteroscedastic":
        center = np.asarray(noise_spec["center"], dtype=float)
        multiplier = _hetero_multiplier(
            X,
            center=center,
            peak_multiplier=float(noise_spec["peak_multiplier"]),
            lengthscale_frac=float(noise_spec["lengthscale_frac"]),
        )
        local_var = (base_std ** 2) * multiplier / float(noise_spec["normalizer"])
        return rng.normal(0.0, np.sqrt(local_var), size=n)

    raise ValueError(f"Unknown noise model: {name}")


def add_noise_to_observations(func, X, noise_spec: typing.Dict[str, typing.Any], rng: np.random.RandomState):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    y_latent = np.asarray(func(X), dtype=float).reshape(-1)
    noise = sample_noise(X, noise_spec=noise_spec, rng=rng)
    y_obs = y_latent + noise
    return y_latent, noise, y_obs


def make_noise_sampler(noise_spec: typing.Dict[str, typing.Any]):
    def _sampler(X, rng=None, **kwargs):
        if rng is None:
            rng = np.random.RandomState(0)
        return sample_noise(X, noise_spec=noise_spec, rng=rng)

    return _sampler


def create_experiment_config(dimensions=[5, 10]):
    configs = {}
    for dim in dimensions:
        n_init = max(2 * dim, 5)
        n_iter = min(10 * dim, 200)
        #n_iter = 50
        acq_samples = 3 * dim
        bounds = np.array([[0, 1] for _ in range(dim)])
        # kernel = (
        #     C(1.0, (1e-3, 1e5))
        #     * Matern(length_scale=np.full(dim, 0.3), length_scale_bounds=(1e-3, 1e3), nu=2.5)
        #     + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e1))
        # )
        # isotropic Matérn: one length scale instead of dim separate ones
        kernel = (
            C(1.0, (1e-3, 1e5))
            * Matern(length_scale=0.3, length_scale_bounds=(1e-3, 1e3), nu=2.5)
            + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e1))
        )
        configs[dim] = {
            "dim": dim,
            "n_init": n_init,
            "n_iter": n_iter,
            "acq_samples": acq_samples,
            "bounds": bounds,
            "kernel": kernel,
        }

    return configs


def run_single_optimization(
    func_name,
    func_info,
    config,
    alpha_config,
    g_config,
    noise_model,
    noise_std,
    seed,
    noise_hyperparams,
):
    func = func_info["func"]
    true_optimum = func_info["optimum"]
    dim = config["dim"]
    n_iter = config["n_iter"]

    if g_config == "step_decrease":
        mid = n_iter // 2
        remainder = n_iter - mid
        g_vec = np.concatenate([np.ones(mid), np.zeros(remainder)])
    else:
        g_vec = np.ones(n_iter) * g_config

    alpha_arg = alpha_config if isinstance(alpha_config, str) else float(alpha_config)

    rng = np.random.RandomState(seed)
    x_init = rng.uniform(0.0, 1.0, size=(config["n_init"], dim))
    noise_spec = build_noise_spec(
        func_name=func_name,
        func_info=func_info,
        dim=dim,
        noise_model=noise_model,
        base_std=noise_std,
        noise_hyperparams=noise_hyperparams,
    )
    _, _, y_init = add_noise_to_observations(
        func,
        x_init,
        noise_spec=noise_spec,
        rng=rng,
    )

    optimizer = bo.BayesianOptimizer(
        func=func,
        kernel=copy.deepcopy(config["kernel"]),
        bounds=config["bounds"],
        g=g_vec,
        alpha=alpha_arg,
        x_init=x_init,
        y_init=y_init,
        n_iter=n_iter,
        acq_samples=config["acq_samples"],
        random_state=seed,
    )

    result = optimizer.simulate_optimization(
        noise_std=noise_std,
        noise_sampler=make_noise_sampler(noise_spec),
        max_val=true_optimum,
        verbose=False,
    )

    final_observed_gap = None
    final_simple_regret = None
    if result["observed_gap_trajectory"] is not None:
        final_observed_gap = float(result["observed_gap_trajectory"][-1])
    if result["simple_regret_trajectory"] is not None:
        final_simple_regret = float(result["simple_regret_trajectory"][-1])

    return {
        "function": func_name,
        "dimension": dim,
        "alpha": str(alpha_config),
        "g": str(g_config),
        "noise_model": noise_model,
        "noise_std": float(noise_std),
        "noise_spec": _serialize_noise_spec(noise_spec),
        "seed": seed,
        "best_so_far_observed": result["best_observed_trajectory"],
        "best_so_far_latent": result["best_latent_trajectory"],
        "observed_gap": result["observed_gap_trajectory"],
        "simple_regret": result["simple_regret_trajectory"],
        "inst_regrets": result["inst_regrets"],
        "cum_regrets": result["cum_regrets"],
        "alpha_history": result["alpha_history"],
        "noise_history": result.get("noise_history"),
        "final_best_observed": result["best_observed"],
        "final_best_latent": result["best_latent"],
        "final_observed_gap": final_observed_gap,
        "final_simple_regret": final_simple_regret,
        "n_evaluations": len(result["best_observed_trajectory"]),
    }


def analyze_results(results):
    df_data = []
    for result in results:
        df_data.append(
            {
                "function": result["function"],
                "dimension": result["dimension"],
                "alpha": result["alpha"],
                "g": result["g"],
                "noise_model": result["noise_model"],
                "noise_std": result["noise_std"],
                "seed": result["seed"],
                "final_best_observed": result["final_best_observed"],
                "final_best_latent": result["final_best_latent"],
                "final_observed_gap": np.nan if result["final_observed_gap"] is None else result["final_observed_gap"],
                "final_simple_regret": np.nan if result["final_simple_regret"] is None else result["final_simple_regret"],
            }
        )

    df = pd.DataFrame(df_data)

    print("\n" + "=" * 90)
    print("MULTI-DIMENSIONAL BENCHMARK RESULTS SUMMARY")
    print("=" * 90)

    if df["final_simple_regret"].notna().any():
        print("\nPerformance by dimension (mean final simple regret; lower is better):")
        print("-" * 80)
        dim_perf = (
            df.dropna(subset=["final_simple_regret"])
            .groupby("dimension")["final_simple_regret"]
            .agg(["mean", "std", "count"])
        )
        for dim, stats in dim_perf.iterrows():
            print(
                f"{dim:>3}D: {stats['mean']:8.4f} ± {stats['std']:6.4f} ({stats['count']} experiments)"
            )

    print("\nPerformance by dimension (mean final latent best value; higher is better):")
    print("-" * 80)
    dim_best = df.groupby("dimension")["final_best_latent"].agg(["mean", "std", "count"])
    for dim, stats in dim_best.iterrows():
        print(
            f"{dim:>3}D: {stats['mean']:8.4f} ± {stats['std']:6.4f} ({stats['count']} experiments)"
        )

    print("\nBest configuration per function-dimension:")
    print("-" * 80)
    for func_name in FUNCTIONS.keys():
        print(f"\n{func_name.upper()}:")
        func_results = df[df["function"] == func_name]
        dims_iter = sorted(func_results["dimension"].unique())
        for dim in dims_iter:
            dim_results = func_results[func_results["dimension"] == dim]
            if dim_results["final_simple_regret"].notna().any():
                grouped = dim_results.groupby(["noise_model", "alpha", "g", "noise_std"])["final_simple_regret"].mean()
                best_config = grouped.idxmin()
                best_value = grouped.min()
                noise_model, alpha, g, noise = best_config
                print(
                    f"  {dim:2}D: noise={noise_model}, α={alpha}, g={g}, σ={noise:.3f} → simple regret {best_value:8.4f}"
                )
            else:
                grouped = dim_results.groupby(["noise_model", "alpha", "g", "noise_std"])["final_best_latent"].mean()
                best_config = grouped.idxmax()
                best_value = grouped.max()
                noise_model, alpha, g, noise = best_config
                print(
                    f"  {dim:2}D: noise={noise_model}, α={alpha}, g={g}, σ={noise:.3f} → latent best {best_value:8.4f}"
                )

    if df["final_simple_regret"].notna().any():
        print("\nFunction difficulty ranking by dimension (harder = larger final simple regret):")
        print("-" * 90)
        for dim in sorted(df["dimension"].unique()):
            dim_results = df[(df["dimension"] == dim) & (df["final_simple_regret"].notna())]
            if len(dim_results) == 0:
                continue
            difficulty = dim_results.groupby("function")["final_simple_regret"].mean().sort_values(ascending=False)
            print(f"\n{dim}D:")
            for rank, (func, score) in enumerate(difficulty.items(), 1):
                print(f"  {rank:>2}. {func:>24}: {score:8.4f}")

    return df



def run_benchmark(
    model_output_dir: pathlib.Path,
    noise_models=None,
    noise_levels=None,
    noise_hyperparams=None,
):
    print("Running Multi-Dimensional Bayesian Optimization Benchmark")
    print("=" * 60)

    if noise_models is None:
        noise_models = list(NOISE_MODELS)
    if noise_levels is None:
        noise_levels = list(NOISE_LEVELS)
    if noise_hyperparams is None:
        noise_hyperparams = {
            "student_t_df": STUDENT_T_DF,
            "mixture_contamination": GAUSSIAN_MIXTURE_CONTAMINATION,
            "mixture_scale": GAUSSIAN_MIXTURE_SCALE,
            "hetero_peak_multiplier": HETERO_PEAK_MULTIPLIER,
            "hetero_lengthscale_frac": HETERO_LENGTHSCALE_FRAC,
            "hetero_mc_samples": HETERO_NORMALIZER_MC_SAMPLES,
        }

    configs = create_experiment_config(DIMENSIONS)
    results = []
    model_output_dir.mkdir(parents=True, exist_ok=True)

    total_experiments = (
        len(FUNCTIONS)
        * len(DIMENSIONS)
        * len(ALPHA_CONFIGS)
        * len(G_CONFIGS)
        * len(noise_models)
        * len(noise_levels)
        * len(SEEDS)
    )


    print(f"Dimensions tested: {DIMENSIONS}")
    print(f"FUNCTIONS: {list(FUNCTIONS.keys())}")
    print(f"Noise models: {noise_models}")
    for noise_model in noise_models:
        for noise_std in noise_levels:
            print("  - " + noise_model_summary(noise_model, float(noise_std), noise_hyperparams))
    print(f"Total experiments: {total_experiments}")
    print()

    experiment_count = 0
    noise_slug = _noise_slug(noise_models)
    print(noise_slug)
    for dim in DIMENSIONS:
        config = configs[dim]
        print(f"\n--- DIMENSION {dim}D ---")

        for func_name, func_info in FUNCTIONS.items():
            print(f"Testing {func_name} ({func_info['description']})...")
            for noise_model in noise_models:
                for alpha in ALPHA_CONFIGS:
                    for g in G_CONFIGS:
                        for noise_std in noise_levels:
                            for seed in SEEDS:
                                experiment_count += 1
                                print(
                                    f"  Exp {experiment_count}/{total_experiments}: noise={noise_model}, α={alpha}, g={g}, σ={noise_std:.3f}, seed={seed}",
                                    end="",
                                )
                                try:
                                    result = run_single_optimization(
                                        func_name,
                                        func_info,
                                        config,
                                        alpha,
                                        g,
                                        noise_model,
                                        noise_std,
                                        seed,
                                        noise_hyperparams,
                                    )
                                    results.append(result)
                                    print(" ✓")
                                except Exception as e:
                                    print(f" ✗ ERROR: {str(e)[:100]}...")
                                    continue

            with open(model_output_dir.joinpath(f"s01_batch1_{noise_slug}_raw_progress_ai_2.pkl"), "wb") as f:
                pickle.dump(results, f)


    df = analyze_results(results)
    df.to_csv(model_output_dir.joinpath(f"s01_batch1_{noise_slug}_experiment_results_ai_2.csv"), index=False)
    with open(model_output_dir.joinpath(f"s01_batch1_{noise_slug}_experiment_results_ai_2.pkl"), "wb") as f:
        pickle.dump(results, f)
    return results


if __name__ == "__main__":
    arguments = parse_args()
    selected_noise_models = get_selected_noise_models(arguments)
    selected_noise_levels = get_selected_noise_levels(arguments)
    noise_hyperparams = get_noise_hyperparameters(arguments)
    print(arguments["model_output_dir"])
    results = run_benchmark(
        model_output_dir=pathlib.Path(arguments["model_output_dir"]),
        noise_models=selected_noise_models,
        noise_levels=selected_noise_levels,
        noise_hyperparams=noise_hyperparams,
    )
