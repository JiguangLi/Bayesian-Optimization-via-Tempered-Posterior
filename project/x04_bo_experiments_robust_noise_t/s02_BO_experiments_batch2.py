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
        default=pathlib.Path.home() / "BO" / "models" / "X04_t",
    )
    parser.add_argument(
        "--noise_model",
        choices=["gaussian", "student_t", "gaussian_mixture", "heteroscedastic"],
        default="student_t",
        help="Override NOISE_MODELS with a single noise model.",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=None,
        help="Override NOISE_LEVELS with a single baseline standard deviation.",
    )
    parser.add_argument("--student_t_df", type=float, default=4.0)
    parser.add_argument("--mixture_contamination", type=float, default=0.10)
    parser.add_argument("--mixture_scale", type=float, default=10.0)
    parser.add_argument("--hetero_peak_multiplier", type=float, default=8.0)
    parser.add_argument("--hetero_lengthscale_frac", type=float, default=0.02)
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
NOISE_MODELS = ["student_t"]
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

def bukin6_2d(x, dim=None):
    """Bukin function N.6 (2D). 2D only
    Domain (native): x1 ∈ [-15, -5], x2 ∈ [-3, 3]
    Global minimum: f(-10, 1) = 0  → after rescaling, argmax at x = (0.5, 2/3)
    Returns negative value so max = 0 at the optimum.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Bukin N.6 is 2D; expected x with 2 columns.")

    # Linear rescaling from [0,1]^2 to native domain
    u1 = -15.0 + 10.0 * x[:, 0]   # [-15, -5]
    u2 = -3.0  +  6.0 * x[:, 1]   # [-3, 3]

    term1 = 100.0 * np.sqrt(np.abs(u2 - 0.01 * u1**2))
    term2 = 0.01 * np.abs(u1 + 10.0)
    value = term1 + term2
    return -value  # maximum 0 at x* = (0.5, 2/3)


def cross_in_tray_2d(x, dim=None):
    """Cross-in-Tray (2D).
    Native domain: u1,u2 ∈ [-10, 10], 2D only
    Native global minima: f(u*) ≈ -2.06261218 at (±1.34941, ±1.34941)
    We rescale from [0,1]^2 and shift so the returned objective has max = 0 at the minima.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Cross-in-Tray is 2D; expected x with 2 columns.")

    # Linear rescaling from [0,1]^2 → [-10,10]^2
    u1 = 20.0 * x[:, 0] - 10.0
    u2 = 20.0 * x[:, 1] - 10.0

    # Native Cross-in-Tray value (negative near minima)
    fact1 = np.sin(u1) * np.sin(u2)
    r = np.sqrt(u1**2 + u2**2)
    fact2 = np.exp(np.abs(100.0 - r / np.pi))
    f_native = -0.0001 * (np.abs(fact1 * fact2) + 1.0) ** 0.1

    # Shift so the minimum becomes 0, then negate → maximum = 0 at the global minimizers
    f_min = -2.06261218  # known global minimum value
    value = f_native - f_min
    return -value

def drop_wave_nd(x, dim=None):
    """Drop-Wave (nD).
    Native domain: u_j ∈ [-5.12, 5.12]
    Native global minimum: f(u)= - (1 + cos(12*||u||)) / (0.5*||u||^2 + 2), minimized at u=0 with f_min = -1.
    We rescale from [0,1]^d and shift so the returned objective has max = 0 at the minimum.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    u = 10.24 * x - 5.12  # linear rescaling [0,1]^d -> [-5.12, 5.12]^d

    r = np.linalg.norm(u, axis=1)
    f_native = - (1.0 + np.cos(12.0 * r)) / (0.5 * r**2 + 2.0)  # ≤ -1, with min -1 at u=0
    value = f_native + 1.0  # shift so min becomes 0
    return -value           # -> maximum 0 at x* = 0.5·1


def eggholder_2d(x, dim=None):
    """Eggholder (2D).
    Native domain: u1,u2 ∈ [-512, 512]
    Native global minimum: f(512, 404.2319) ≈ -959.6406627
    We rescale from [0,1]^2 and shift so the returned objective has max = 0 at the minimum.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Eggholder is 2D; expected x with 2 columns.")

    # Linear rescaling [0,1]^2 → [-512, 512]^2
    u1 = 1024.0 * x[:, 0] - 512.0
    u2 = 1024.0 * x[:, 1] - 512.0

    term1 = -(u2 + 47.0) * np.sin(np.sqrt(np.abs(u2 + (u1 / 2.0) + 47.0)))
    term2 = -u1 * np.sin(np.sqrt(np.abs(u1 - (u2 + 47.0))))
    f_native = term1 + term2

    # Shift so the minimum becomes 0, then negate → maximum = 0 at the global minimizer
    f_min = -959.640662711  # known global minimum value
    value = f_native - f_min
    return -value  # argmax occurs at x* ≈ ((512+512)/1024, (404.2319+512)/1024) ≈ (1.0, 0.895)


def grlee12_nd(x, dim=None):
    """Gramacy & Lee (2012) — nD separable extension.
    Each coordinate u_j ∈ [0.5, 2.5], f_nd(u) = Σ_j [sin(10πu_j)/(2u_j) + (u_j-1)^4]
    The per-dimension minimum is f_min_1d; global minimum is d * f_min_1d at u_j = u* ∀j.
    Returns: -(f_nd(u) - d*f_min_1d) so the returned maximum is 0 at the minimizer.
    ND status: The canonical benchmark is 1D; this is a common separable nD extension.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Linear rescaling [0,1]^d -> [0.5, 2.5]^d
    u = 2.0 * x + 0.5  # shape (n, d)
    # Evaluate per-dim and sum
    f_per_dim = np.sin(10.0 * np.pi * u) / (2.0 * u) + (u - 1.0) ** 4
    f_native = np.sum(f_per_dim, axis=1)

    d = u.shape[1]
    f_min_1d = -0.8690111349895001
    value = f_native - d * f_min_1d
    return -value  # maximum 0 at x* = ((u* - 0.5)/2) * 1_d  (~0.02428 on each coord)

def holder_table_2d(x, dim=None):
    """Hölder Table (2D).
    Native domain: u1,u2 ∈ [-10, 10]
    Native global minima: f(u*) ≈ -19.20850257 at (±8.05502, ±9.66459) (four points)
    We rescale from [0,1]^2 and shift so the returned objective has max = 0 at the minima.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Hölder Table is 2D; expected x with 2 columns.")

    # Linear rescaling [0,1]^2 → [-10, 10]^2
    u1 = 20.0 * x[:, 0] - 10.0
    u2 = 20.0 * x[:, 1] - 10.0

    fact1 = np.sin(u1) * np.cos(u2)
    r = np.sqrt(u1**2 + u2**2)
    fact2 = np.exp(np.abs(1.0 - r / np.pi))
    f_native = -np.abs(fact1 * fact2)

    # Shift so the minimum becomes 0, then negate → maximum = 0 at the global minimizers
    f_min = -19.2085025679
    value = f_native - f_min
    return -value

    

def levy13_2d(x, dim=None):
    """Levy N.13 (2D, canonical).
    Native domain: u1,u2 ∈ [-10, 10]
    Global minimum: f(1,1) = 0
    Returns: -f(u) so the maximum is 0 at the minimizer.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Levy N.13 is 2D; expected x with 2 columns.")

    # Linear rescaling [0,1]^2 → [-10,10]^2
    u1 = 20.0 * x[:, 0] - 10.0
    u2 = 20.0 * x[:, 1] - 10.0

    term1 = np.sin(3.0 * np.pi * u1) ** 2
    term2 = (u1 - 1.0) ** 2 * (1.0 + np.sin(3.0 * np.pi * u2) ** 2)
    term3 = (u2 - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * u2) ** 2)
    f = term1 + term2 + term3
    return -f  # max 0 at (u1,u2)=(1,1) → x*=(0.55, 0.55)


def schaffer2_2d(x, dim=None):
    """Schaffer N.2 (2D, canonical).
    Native domain: u1,u2 ∈ [-100, 100]
    f(u) = 0.5 + (sin^2(u1^2 - u2^2) - 0.5) / (1 + 0.001*(u1^2 + u2^2))^2
    Global minimum: f(0,0) = 0
    Returns: -f(u) so the maximum is 0 at the minimizer.
    ND status: 2D ONLY (no standard n-D), see schaffer2_nd for an expanded n-D version.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Schaffer N.2 is 2D; expected x with 2 columns.")

    # Linear rescaling [0,1]^2 → [-100, 100]^2
    u1 = 200.0 * x[:, 0] - 100.0
    u2 = 200.0 * x[:, 1] - 100.0

    num = np.sin(u1**2 - u2**2)**2 - 0.5
    den = (1.0 + 0.001 * (u1**2 + u2**2))**2
    f_native = 0.5 + num / den
    return -f_native  # max 0 at x* = (0.5, 0.5)

def schaffer4_2d(x, dim=None):
    """Schaffer N.4 (2D).
    Native domain: u1,u2 ∈ [-100, 100]
    f(u) = 0.5 + (cos^2(sin(|u1^2 - u2^2|)) - 0.5) / (1 + 0.001*(u1^2 + u2^2))**2
    Global minimum ≈ 0.292579 (various symmetric locations).
    Returns: -(f(u) - f_min) so the returned maximum is 0 at the minimizer.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Schaffer N.4 is 2D; expected x with 2 columns.")

    # Linear rescaling [0,1]^2 → [-100, 100]^2
    u1 = 200.0 * x[:, 0] - 100.0
    u2 = 200.0 * x[:, 1] - 100.0

    num = (np.cos(np.sin(np.abs(u1**2 - u2**2)))**2) - 0.5
    den = (1.0 + 0.001 * (u1**2 + u2**2))**2
    f_native = 0.5 + num / den

    f_min = 0.292579  # known global minimum (approx.)
    return -(f_native - f_min)


def shubert_2d(x, dim=None):
    """Shubert function (2D).
    Native domain: u1,u2 ∈ [-10, 10]
    f(u) = [Σ_{i=1}^5 i cos((i+1)u1 + i)] * [Σ_{i=1}^5 i cos((i+1)u2 + i)]
    Known global minimum ≈ -186.7309088 at multiple points.
    Returns: -(f(u) - f_min) so the returned maximum is 0 at any global minimizer.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Shubert is 2D; expected x with 2 columns.")

    # Linear rescaling [0,1]^2 → [-10, 10]^2
    u1 = 20.0 * x[:, 0] - 10.0
    u2 = 20.0 * x[:, 1] - 10.0

    i = np.arange(1, 6, dtype=float)                 # 1..5
    sum1 = np.sum(i * np.cos((i + 1) * u1[:, None] + i), axis=1)
    sum2 = np.sum(i * np.cos((i + 1) * u2[:, None] + i), axis=1)
    f_native = sum1 * sum2

    f_min = -186.7309088  # widely reported global minimum
    return -(f_native - f_min)  # => max = 0 at global minima

def bohachevsky1_2d(x, dim=None):
    """Bohachevsky Function 1 (2D).
    Native domain: u1,u2 ∈ [-100, 100]
    f(u) = u1^2 + 2 u2^2 - 0.3 cos(3π u1) - 0.4 cos(4π u2) + 0.7
    Global minimum: f(0,0) = 0  → after negation, returned max is 0 at x* = (0.5, 0.5).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Bohachevsky 1 is 2D; expected x with 2 columns.")

    # Rescale [0,1]^2 → [-100, 100]^2
    u1 = 200.0 * x[:, 0] - 100.0
    u2 = 200.0 * x[:, 1] - 100.0

    f_native = (u1**2
                + 2.0 * (u2**2)
                - 0.3 * np.cos(3.0 * np.pi * u1)
                - 0.4 * np.cos(4.0 * np.pi * u2)
                + 0.7)

    return -f_native  # maximize this; max = 0 at (u1,u2)=(0,0) → x=(0.5,0.5)



def rotated_hyper_ellipsoid_nd(x, dim=None):
    """Rotated Hyper-Ellipsoid (nD).
    Native domain: u_j ∈ [-65.536, 65.536]
    f(u) = Σ_{i=1}^d Σ_{j=1}^i u_j^2  (a.k.a. cumulative sum of squares)
    Global minimum: f(0)=0  → after negation, returned max is 0 at x* = 0.5·1_d.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Linear rescaling [0,1]^d → [-65.536, 65.536]^d
    u = 131.072 * x - 65.536

    # f(u) = sum over i of prefix sums of u^2
    prefix = np.cumsum(u**2, axis=1)
    value = np.sum(prefix, axis=1)
    return -value


def sum_squares_nd(x, dim=None):
    """Sum Squares (nD).
    Native domain: u_j ∈ [-10, 10]
    f(u) = Σ_{j=1}^d j * u_j^2
    Global minimum: f(0)=0 → after negation, returned max is 0 at x* = 0.5·1_d.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Linear rescaling [0,1]^d → [-10, 10]^d
    u = 20.0 * x - 10.0
    d = u.shape[1]

    i = np.arange(1, d + 1, dtype=float)  # weights 1..d
    value = np.sum(i * (u**2), axis=1)
    return -value

def booth_2d(x, dim=None):
    """Booth function (2D).
    Native domain: u1,u2 ∈ [-10, 10]
    f(u) = (u1 + 2u2 - 7)^2 + (2u1 + u2 - 5)^2
    Global minimum: f(1,3) = 0  → after negation, returned max is 0 at x* = ((1+10)/20, (3+10)/20) = (0.55, 0.65).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Booth is 2D; expected x with 2 columns.")

    # Rescale [0,1]^2 → [-10, 10]^2
    u1 = 20.0 * x[:, 0] - 10.0
    u2 = 20.0 * x[:, 1] - 10.0

    f_native = (u1 + 2.0 * u2 - 7.0) ** 2 + (2.0 * u1 + u2 - 5.0) ** 2
    return -f_native  # maximize this; max = 0 at (u1,u2)=(1,3) → x=(0.55,0.65)


def matyas_2d(x, dim=None):
    """Matyas function (2D).
    Native domain: u1,u2 ∈ [-10, 10]
    f(u) = 0.26*(u1^2 + u2^2) - 0.48*u1*u2
    Global minimum: f(0,0)=0 → after negation, returned max is 0 at x* = (0.5, 0.5).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Matyas is 2D; expected x with 2 columns.")

    # Rescale [0,1]^2 → [-10, 10]^2
    u1 = 20.0 * x[:, 0] - 10.0
    u2 = 20.0 * x[:, 1] - 10.0

    f_native = 0.26 * (u1**2 + u2**2) - 0.48 * u1 * u2
    return -f_native  # maximize this; max = 0 at (u1,u2)=(0,0) → x=(0.5,0.5)


def mccormick_2d(x, dim=None):
    """McCormick function (2D).
    Native domain: u1 ∈ [-1.5, 4], u2 ∈ [-3, 4]
    f(u) = sin(u1 + u2) + (u1 - u2)^2 - 1.5*u1 + 2.5*u2 + 1
    Known global minimum ≈ -1.913222954981 at (u1,u2) ≈ (-0.54719, -1.54719).
    Returns: -(f(u) - f_min) so the returned maximum is 0 at the minimizer.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("McCormick is 2D; expected x with 2 columns.")

    # Linear rescaling [0,1]^2 → [-1.5,4] × [-3,4]
    u1 = -1.5 + 5.5 * x[:, 0]  # [-1.5, 4]
    u2 = -3.0 + 7.0 * x[:, 1]  # [-3, 4]

    f_native = (
        np.sin(u1 + u2)
        + (u1 - u2) ** 2
        - 1.5 * u1
        + 2.5 * u2
        + 1.0
    )

    f_min = -1.913222954981  # reported global minimum
    return -(f_native - f_min)  # ⇒ max = 0 at the global minimizer

def power_sum_4d(x, dim=None, b=(8.0, 18.0, 44.0, 114.0)):
    """Power Sum (fixed 4D).
    Input: x ∈ [0,1]^4  → rescale to u = 4 * x ∈ [0,4]^4
    f(u) = Σ_{i=1}^4 ( Σ_{j=1}^4 u_j^i  -  b_i )^2
    Returns: -f(u) so the maximum is 0 when the moment constraints are met.

    Notes
    -----
    - Fixed dimensionality: 4D only.
    - 'dim' is accepted for API consistency; if provided, it must be 4.
    - 'b' defaults to the common benchmark vector (8, 18, 44, 114).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n, d = x.shape
    if d != 4:
        raise ValueError(f"power_sum_4d expects 4 columns, got d={d}")
    if dim is not None and dim != 4:
        raise ValueError(f"dim must be 4 for power_sum_4d, got dim={dim}")

    b_vec = np.asarray(b, dtype=float)
    if b_vec.shape != (4,):
        raise ValueError("b must be length-4 (default: (8, 18, 44, 114))")

    # Rescale [0,1]^4 → [0,4]^4
    u = 4.0 * x  # (n, 4)

    # inner_i = sum_j u_j^i, for i = 1..4
    i = np.arange(1, 5, dtype=float)            # (4,)
    u_pow = u[:, None, :] ** i[:, None]         # (n, 4, 4) axis=1 indexes i
    inner = np.sum(u_pow, axis=2)               # (n, 4)

    diff = inner - b_vec[None, :]               # (n, 4)
    value = np.sum(diff**2, axis=1)             # (n,)

    return -value  # max = 0 when inner == b



FUNCTIONS_Batch2 = {
    'bukin6': {
        'func': bukin6_2d,
        'optimum': 0.0,
        'description': '2D only; steep valley with ridges',
        'dim': 2,
    },
    'cross_in_tray': {
        'func': cross_in_tray_2d,
        'optimum': 0.0,
        'description': '2D only; highly multimodal with 4 symmetric minima',
        'dim': 2,
    },
    'drop_wave': {
        'func': drop_wave_nd,
        'optimum': 0.0,
        'description': 'Radial ripples; many local minima (nD)',
        'dim': None,
    },
    'eggholder': {
        'func': eggholder_2d,
        'optimum': 0.0,
        'description': '2D only; very rugged landscape',
        'dim': 2,
    },
    'grlee12': {
        'func': grlee12_nd,
        'optimum': 0.0,
        'description': 'Separable oscillatory extension of GL(2012); sharp around u≈0.55 (nD)',
        'dim': None,
    },
    'holder_table': {
        'func': holder_table_2d,
        'optimum': 0.0,
        'description': '2D only; rugged with four symmetric minima',
        'dim': 2,
    },
    'levy13': {
        'func': levy13_2d,
        'optimum': 0.0,
        'description': '2D only; multimodal, minimum at (1,1)',
        'dim': 2,
    },
    'schaffer2': {
        'func': schaffer2_2d,
        'optimum': 0.0,
        'description': '2D only; oscillatory cliffs, minimum at origin',
        'dim': 2,
    },
    'schaffer4': {
        'func': schaffer4_2d,
        'optimum': 0.0,
        'description': '2D only; oscillatory, shallow basin (shifted)',
        'dim': 2,
    },
    'shubert': {
        'func': shubert_2d,
        'optimum': 0.0,
        'description': '2D only; many symmetric minima',
        'dim': 2,
    },
    'bohachevsky1': {
        'func': bohachevsky1_2d,
        'optimum': 0.0,
        'description': '2D only; quadratic with cosine ripples',
        'dim': 2,
    },
    'rotated_hyper_ellipsoid': {
        'func': rotated_hyper_ellipsoid_nd,
        'optimum': 0.0,
        'description': 'Non-separable cumulative sum of squares (nD)',
        'dim': None,
    },
    'sum_squares': {
        'func': sum_squares_nd,
        'optimum': 0.0,
        'description': 'Separable quadratic with increasing weights (nD)',
        'dim': None,
    },
    'booth': {
        'func': booth_2d,
        'optimum': 0.0,
        'description': '2D only; quadratic with unique minimum at (1,3)',
        'dim': 2,
    },
    'matyas': {
        'func': matyas_2d,
        'optimum': 0.0,
        'description': '2D only; quadratic with cross term',
        'dim': 2,
    },
    'mccormick': {
        'func': mccormick_2d,
        'optimum': 0.0,
        'description': '2D only; non-convex with curved valley',
        'dim': 2,
    },
    'power_sum': {
        'func': power_sum_4d,
        'optimum': 0.0,
        'description': 'Moment-matching objective parameterized by b (nD)',
        'dim': 4,
    },
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
    unique_dims = sorted({cfg["dim"] for cfg in FUNCTIONS_Batch2.values() if cfg["dim"] is not None})
    all_dims = list(dimensions) + [d for d in unique_dims if d not in dimensions]
    for dim in all_dims:
        n_init = max(2 * dim, 5)
        n_iter = min(10 * dim, 200)
        #n_iter = 50
        acq_samples = 3 * dim
        bounds = np.array([[0, 1] for _ in range(dim)])
        # kernel = (
        #     C(1.0, (1e-3, 1e3))
        #     * Matern(length_scale=np.full(dim, 0.3), length_scale_bounds=(1e-3, 100), nu=2.5)
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
    for func_name in FUNCTIONS_Batch2.keys():
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



def dims_for_func(func_info, DIMENSIONS):
    return [func_info["dim"]] if func_info.get("dim") is not None else list(DIMENSIONS)


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
        sum(len(dims_for_func(func_info, DIMENSIONS)) for func_info in FUNCTIONS_Batch2.values())
        * len(ALPHA_CONFIGS)
        * len(G_CONFIGS)
        * len(noise_models)
        * len(noise_levels)
        * len(SEEDS)
    )


    print(f"Dimensions tested: {DIMENSIONS}")
    print(f"FUNCTIONS_Batch2: {list(FUNCTIONS_Batch2.keys())}")
    print(f"Noise models: {noise_models}")
    for noise_model in noise_models:
        for noise_std in noise_levels:
            print("  - " + noise_model_summary(noise_model, float(noise_std), noise_hyperparams))
    print(f"Total experiments: {total_experiments}")
    print()

    experiment_count = 0
    noise_slug = _noise_slug(noise_models)

    for func_name, func_info in FUNCTIONS_Batch2.items():
        for dim in dims_for_func(func_info, DIMENSIONS):
            if dim not in configs:
                print(f"[skip] {func_name}: no config for dim={dim}")
                continue

            config = configs[dim]
            print(f"\n--- {func_name} | {dim}D ---")
            print(f"{func_info['description']}")

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

            with open(model_output_dir.joinpath(f"s02_batch2_{noise_slug}_raw_progress_ai_0.01.pkl"), "wb") as f:
                pickle.dump(results, f)


    df = analyze_results(results)
    df.to_csv(model_output_dir.joinpath(f"s02_batch2_{noise_slug}_experiment_results_ai_2.csv"), index=False)
    with open(model_output_dir.joinpath(f"s02_batch2_{noise_slug}_experiment_results_ai_2.pkl"), "wb") as f:
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
