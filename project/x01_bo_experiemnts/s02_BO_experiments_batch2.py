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
DIMENSIONS = [5,10]


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


def create_experiment_config(dimensions=[2, 5, 10]):
    """Create experiment configuration for different dimensions"""
    
    configs = {}
    unique_dims = sorted({cfg['dim'] for cfg in FUNCTIONS_Batch2.values() if cfg['dim'] is not None})
    all_dims = dimensions + unique_dims
    for dim in all_dims:
        # Scale parameters with dimension
        n_init = max(2*dim, 5)
        n_iter = min(10*dim, 200) #max(20, 5 * dim)  # Scale iterations with dimension
        n_iter = max(n_iter, 30) # at least 30 iterations
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

    # handle unknown optimum case
    if true_optimum is None:
        regret = None
        final_regret = None
    else:
        regret = true_optimum - best_so_far
        final_regret = true_optimum - result['best_observed']
    
    return {
        'function': func_name,
        'dimension': dim,
        'alpha': str(alpha_config),
        'g': str(g_config),
        'noise_std': noise_std,
        'seed': seed,
        'best_so_far': best_so_far,
        'final_best': result['best_observed'],
        'regret': regret,
        'final_regret': final_regret,
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
    
    for func_name in FUNCTIONS_Batch2.keys():
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

def dims_for_func(func_info, DIMENSIONS):
    return [func_info['dim']] if func_info.get('dim') is not None else list(DIMENSIONS)


def run_benchmark(model_output_dir: pathlib.Path):
    """Run complete multi-dimensional benchmark suite"""
    
    print("Running Multi-Dimensional Bayesian Optimization Benchmark")
    print("="*60)
    
    # Create configurations for different dimensions
    configs = create_experiment_config(DIMENSIONS)
    
    results = []
    total_experiments = (len(FUNCTIONS_Batch2) * len(DIMENSIONS) * len(ALPHA_CONFIGS) * 
                        len(G_CONFIGS) * len(NOISE_LEVELS) * len(SEEDS))
    
    print(f"Dimensions tested: {DIMENSIONS}")
    print(f"FUNCTIONS_Batch2: {list(FUNCTIONS_Batch2.keys())}")
    print(f"Total experiments (upper bound): {total_experiments}")
    print()
    
    experiment_count = 0


    for func_name, func_info in FUNCTIONS_Batch2.items():
        for dim in dims_for_func(func_info, DIMENSIONS):
            if dim not in configs:
                print(f"[skip] {func_name}: no config for dim={dim}")
                continue

            config = configs[dim]
            print(f"\n--- {func_name} | {dim}D ---")
            print(f"{func_info['description']}")

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
                                print(f" ✗ ERROR: {str(e)[:80]}...")
                                continue
        
    # save result for each dimension  
            with open(model_output_dir.joinpath('s02_batch2_raw_experiment_results.pkl'), 'wb') as f:
                pickle.dump(results, f)
    df = analyze_results(results)
    df.to_csv(model_output_dir.joinpath('s02_batch2_df_experiment_results.csv'))
    
    return results

if __name__ == "__main__":
    # data setup
    arguments = parse_args()
    print(arguments["model_output_dir"])
    results = run_benchmark(model_output_dir = arguments["model_output_dir"])
   



    
    

