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

def camel3_2d(x, dim=None):
    """Three-Hump Camel (2D).
    Native domain: u1,u2 ∈ [-5, 5]
    f(u) = 2u1^2 - 1.05u1^4 + (u1^6)/6 + u1*u2 + u2^2
    Global minimum: f(0,0) = 0  → after negation, returned max is 0 at x* = (0.5, 0.5).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Three-Hump Camel is 2D; expected x with 2 columns.")

    # Rescale [0,1]^2 → [-5, 5]^2
    u1 = 10.0 * x[:, 0] - 5.0
    u2 = 10.0 * x[:, 1] - 5.0

    f_native = 2.0 * u1**2 - 1.05 * u1**4 + (u1**6) / 6.0 + u1 * u2 + u2**2
    return -f_native


def camel6_2d(x, dim=None):
    """Six-Hump Camel (2D).
    Native domain: u1 ∈ [-3, 3], u2 ∈ [-2, 2]
    f(u) = (4 - 2.1*u1^2 + (u1^4)/3) * u1^2 + u1*u2 + (-4 + 4*u2^2) * u2^2
    Global minima ≈ f(u*) = -1.03162845349 at (±0.089842, ∓0.712656).
    Returns: -(f(u) - f_min) so the returned maximum is 0 at the minimizers.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Six-Hump Camel is 2D; expected x with 2 columns.")

    # Linear rescaling [0,1]^2 → [-3,3] × [-2,2]
    u1 = 6.0 * x[:, 0] - 3.0   # [-3, 3]
    u2 = 4.0 * x[:, 1] - 2.0   # [-2, 2]

    f_native = ((4.0 - 2.1 * u1**2 + (u1**4) / 3.0) * u1**2
                + u1 * u2
                + (-4.0 + 4.0 * u2**2) * u2**2)

    f_min = -1.0316284534898772  # known global minimum
    return -(f_native - f_min)



def dixon_price_nd(x, dim=None):
    """Dixon–Price (nD).
    Native domain: u_j ∈ [-10, 10]
    f(u) = (u1 - 1)^2 + Σ_{i=2}^d i * (2*u_i^2 - u_{i-1})^2
    Global minimum: f(u*) = 0 (achieved at a known closed-form u*); we return -f(u) so max = 0.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Rescale [0,1]^d → [-10,10]^d
    u = 20.0 * x - 10.0
    d = u.shape[1]

    term1 = (u[:, 0] - 1.0) ** 2
    if d == 1:
        value = term1
    else:
        i = np.arange(2, d + 1, dtype=float)  # 2..d
        value = term1 + np.sum(i * (2.0 * u[:, 1:]**2 - u[:, :-1])**2, axis=1)
    return -value

def dejong5_2d(x, dim=None):
    """De Jong function N.5 (2D, 'Shekel's Foxholes').
    Native domain: u1,u2 ∈ [-65.536, 65.536]
    f(u) = 1 / ( 0.002 + Σ_{i=1}^{25} 1 / ( i + (u1 - A1_i)^6 + (u2 - A2_i)^6 ) )
    Global minimum ≈ 0.9980038378 at u* ≈ (-32, -32) and other grid points.
    Returns: -(f(u) - f_min) so the returned maximum is 0 at the minimizer.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("De Jong N.5 is 2D; expected x with 2 columns.")

    # Rescale [0,1]^2 → [-65.536, 65.536]^2
    u1 = 131.072 * x[:, 0] - 65.536
    u2 = 131.072 * x[:, 1] - 65.536

    # Grid A (2 x 25) with coordinates from {-32,-16,0,16,32}
    a = np.array([-32., -16., 0., 16., 32.])
    A1 = np.repeat(a, 5)  # shape (25,)
    A2 = np.tile(a, 5)    # shape (25,)

    i = np.arange(1, 26, dtype=float)  # 1..25

    # Compute Σ 1 / (i + (u1 - A1_i)^6 + (u2 - A2_i)^6) per sample
    du1 = (u1[:, None] - A1[None, :]) ** 6
    du2 = (u2[:, None] - A2[None, :]) ** 6
    denom_inner = i[None, :] + du1 + du2
    S = np.sum(1.0 / denom_inner, axis=1)

    f_native = 1.0 / (0.002 + S)

    # Known global minimum value (literature)
    f_min = 0.998003837794  # ~ at (-32, -32) etc.
    return -(f_native - f_min)


def easom_2d(x, dim=None):
    """Easom (2D).
    Native domain: u1,u2 ∈ [-100, 100]
    f(u) = -cos(u1)*cos(u2)*exp(-((u1-π)^2 + (u2-π)^2))
    Global minimum: f(π, π) = -1.
    We return -(f + 1) so the returned maximum is 0 at (π, π).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Easom is 2D; expected x with 2 columns.")

    # Rescale [0,1]^2 → [-100, 100]^2
    u1 = 200.0 * x[:, 0] - 100.0
    u2 = 200.0 * x[:, 1] - 100.0

    f_native = -np.cos(u1) * np.cos(u2) * np.exp(-((u1 - np.pi)**2 + (u2 - np.pi)**2))
    value = f_native + 1.0  # shift so min becomes 0
    return -value           # => max = 0 at (u1,u2) = (π, π)


def michalewicz_nd(x, dim=None, m=10, f_min=None):
    """Michalewicz (nD).
    Native domain: u_j ∈ [0, π]
    f(u) = - Σ_{j=1}^d [ sin(u_j) * ( sin(j * u_j^2 / π) )^(2m) ]
    The landscape becomes more rugged with larger m.
    If f_min is provided (known global minimum value), returns -(f - f_min) so max = 0 at the minimizer.
    Otherwise returns -f(u) (still maximization, but not anchored to 0).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Rescale [0,1]^d → [0, π]^d
    u = np.pi * x
    d = u.shape[1]
    j = np.arange(1, d + 1, dtype=float)[None, :]   # shape (1, d) for broadcasting

    terms = np.sin(u) * (np.sin((j * (u**2)) / np.pi)) ** (2.0 * m)
    f_native = -np.sum(terms, axis=1)

    if f_min is not None:
        return -(f_native - f_min)  # max = 0 at global minimizer
    return -f_native


def beale_2d(x, dim=None):
    """Beale function (2D).
    Native domain: u1,u2 ∈ [-4.5, 4.5]
    f(u) = (1.5 - u1 + u1*u2)^2 + (2.25 - u1 + u1*u2^2)^2 + (2.625 - u1 + u1*u2^3)^2
    Global minimum: f(3, 0.5) = 0  → after negation, returned max is 0 at x* = ((3+4.5)/9, (0.5+4.5)/9).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Beale is 2D; expected x with 2 columns.")

    # Rescale [0,1]^2 → [-4.5, 4.5]^2
    u1 = 9.0 * x[:, 0] - 4.5
    u2 = 9.0 * x[:, 1] - 4.5

    f_native = ((1.5   - u1 + u1 * u2)    ** 2 +
                (2.25  - u1 + u1 * u2**2) ** 2 +
                (2.625 - u1 + u1 * u2**3) ** 2)

    return -f_native


def branin_2d(x, dim=None, a=1.0, b=5.1/(4*np.pi**2), c=5/np.pi, r=6.0, s=10.0, t=1/(8*np.pi)):
    """Branin (2D).
    Native domain: u1 ∈ [-5, 10], u2 ∈ [0, 15]
    f(u) = a*(u2 - b*u1^2 + c*u1 - r)^2 + s*(1 - t)*cos(u1) + s
    Global minima value (standard params): f_min ≈ 0.39788735772973816 at three points.
    Returns: -(f(u) - f_min) so the returned maximum is 0 at any global minimizer.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Branin is 2D; expected x with 2 columns.")

    # Linear rescaling [0,1]^2 → [-5,10] × [0,15]
    u1 = 15.0 * x[:, 0] - 5.0
    u2 = 15.0 * x[:, 1] - 0.0

    f_native = a * (u2 - b * u1**2 + c * u1 - r)**2 + s * (1.0 - t) * np.cos(u1) + s

    # Standard-parameter global minimum
    f_min = 0.39788735772973816
    return -(f_native - f_min)

def colville_4d(x, dim=None):
    """Colville function (4D).
    Native domain: u1..u4 ∈ [-10, 10]
    f(u) = 100(u1^2 - u2)^2 + (u1 - 1)^2 + (u3 - 1)^2
           + 90(u3^2 - u4)^2 + 10.1[(u2 - 1)^2 + (u4 - 1)^2]
           + 19.8(u2 - 1)(u4 - 1)
    Global minimum: f(1,1,1,1) = 0 → after negation, returned max is 0 at x* = (0.55,0.55,0.55,0.55).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 4:
        raise ValueError("Colville is 4D; expected x with 4 columns.")

    # Rescale [0,1]^4 → [-10, 10]^4
    u = 20.0 * x - 10.0
    u1, u2, u3, u4 = u[:, 0], u[:, 1], u[:, 2], u[:, 3]

    f_native = (
        100.0 * (u1**2 - u2)**2
        + (u1 - 1.0)**2
        + (u3 - 1.0)**2
        + 90.0 * (u3**2 - u4)**2
        + 10.1 * ((u2 - 1.0)**2 + (u4 - 1.0)**2)
        + 19.8 * (u2 - 1.0) * (u4 - 1.0)
    )
    return -f_native

def forretal08_1d(x, dim=None):
    """Forrester et al. (2008) — 1D.
    Native domain: u ∈ [0, 1]
    f(u) = (6u - 2)^2 * sin(12u - 4)
    Global minimum (this variant): f_min ≈ -6.020740055735768 at u* ≈ 0.757249
    Returns: -(f(u) - f_min) so the returned maximum is 0 at the minimizer.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 1:
        raise ValueError("Forrester 2008 is 1D; expected x with 1 column.")

    # Rescale [0,1] → [0,1] (identity, shown for consistency)
    u = x[:, 0]

    f_native = (6.0 * u - 2.0) ** 2 * np.sin(12.0 * u - 4.0)

    f_min = -6.020740055735768  # numeric (global) minimum on [0,1]
    return -(f_native - f_min)

def goldstein_price_2d(x, dim=None):
    """Goldstein–Price (2D).
    Native domain: u1,u2 ∈ [-2, 2]
    Global minimum: f(0, -1) = 3
    Returns: -(f(u) - 3) so the returned maximum is 0 at the minimizer.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 2:
        raise ValueError("Goldstein–Price is 2D; expected x with 2 columns.")

    # Rescale [0,1]^2 → [-2, 2]^2
    u1 = 4.0 * x[:, 0] - 2.0
    u2 = 4.0 * x[:, 1] - 2.0

    fact1a = (u1 + u2 + 1.0) ** 2
    fact1b = 19.0 - 14.0*u1 + 3.0*u1**2 - 14.0*u2 + 6.0*u1*u2 + 3.0*u2**2
    fact1  = 1.0 + fact1a * fact1b

    fact2a = (2.0*u1 - 3.0*u2) ** 2
    fact2b = 18.0 - 32.0*u1 + 12.0*u1**2 + 48.0*u2 - 36.0*u1*u2 + 27.0*u2**2
    fact2  = 30.0 + fact2a * fact2b

    f_native = fact1 * fact2
    f_min = 3.0
    return -(f_native - f_min)


def hartmann3_3d(x, dim=None, f_min=None):
    """Hartmann 3D (fixed 3D).
    Domain: u ∈ [0,1]^3
    Native form: f(u) = - Σ_{k=1}^4 α_k * exp( - Σ_{j=1}^3 A[k,j] * (u_j - P[k,j])^2 )
    If f_min is provided, returns -(f - f_min) so the maximum is 0 at the global minimizer.
    Otherwise returns -f(u).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 3:
        raise ValueError("Hartmann-3D is fixed 3D; expected x with 3 columns.")

    # Native parameters
    alpha = np.array([1.0, 1.2, 3.0, 3.2], dtype=float)
    A = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0]
    ], dtype=float)
    P = 1e-4 * np.array([
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [ 381, 5743, 8828]
    ], dtype=float)

    # u = x in this case (domain is [0,1]^3)
    u = x

    # inner_k = sum_j A[k,j] * (u_j - P[k,j])^2
    diff = u[:, None, :] - P[None, :, :]          # (n, 4, 3)
    inner = np.sum(A[None, :, :] * diff**2, axis=2)  # (n, 4)

    f_native = -np.sum(alpha[None, :] * np.exp(-inner), axis=1)  # (n,)

    if f_min is not None:
        return -(f_native - f_min)  # anchor so max=0 at the minimizer
    return -f_native

def hartmann4_4d(x, dim=None, f_min=None):
    """Hartmann 4D (fixed 4D).
    Domain: u ∈ [0,1]^4 (no rescaling needed)
    Native form from SFU:
        alpha = [1.0, 1.2, 3.0, 3.2]
        A (4x6), P (4x6); only first 4 columns are used.
        outer = sum_k alpha[k] * exp( - sum_j A[k,j] * (u_j - P[k,j])^2 )
        f_native = (1.1 - outer) / 0.839
    If f_min is provided, returns -(f_native - f_min) to anchor max=0 at the minimizer.
    Otherwise returns -f_native (unanchored; optimum unknown in your registry).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 4:
        raise ValueError("Hartmann-4D is fixed 4D; expected x with 4 columns.")

    alpha = np.array([1.0, 1.2, 3.0, 3.2], dtype=float)

    A = np.array([
        [10.0,  3.0, 17.0,  3.5,  1.7,  8.0],
        [ 0.05,10.0, 17.0,  0.1,  8.0, 14.0],
        [ 3.0,  3.5,  1.7, 10.0, 17.0,  8.0],
        [17.0,  8.0,  0.05,10.0,  0.1, 14.0]
    ], dtype=float)

    P = 1e-4 * np.array([
        [1312, 1696, 5569,  124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091,  381]
    ], dtype=float)

    A4 = A[:, :4]  # use first 4 columns
    P4 = P[:, :4]

    u = x  # domain already [0,1]^4

    # inner_k = sum_j A[k,j] * (u_j - P[k,j])^2  for k=1..4
    diff = u[:, None, :] - P4[None, :, :]          # (n, 4, 4)
    inner = np.sum(A4[None, :, :] * diff**2, axis=2)  # (n, 4)

    outer = np.sum(alpha[None, :] * np.exp(-inner), axis=1)
    f_native = (1.1 - outer) / 0.839

    if f_min is not None:
        return -(f_native - f_min)   # anchored: max = 0 at minimizer
    return -f_native              


def hartmann6_6d(x, dim=None, f_min=None):
    """Hartmann 6D (fixed 6D).
    Domain: u ∈ [0,1]^6 (no rescaling)
    Native SFU form:
        alpha = [1.0, 1.2, 3.0, 3.2]
        A (4x6), P (4x6)
        outer = sum_k alpha[k] * exp( - sum_j A[k,j] * (u_j - P[k,j])^2 )
        f_native = -(2.58 + outer) / 1.94
    If f_min is provided, returns -(f_native - f_min) so max = 0 at the minimizer.
    Otherwise returns -f_native (unanchored; treat optimum as unknown in your registry).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 6:
        raise ValueError("Hartmann-6D is fixed 6D; expected x with 6 columns.")

    alpha = np.array([1.0, 1.2, 3.0, 3.2], dtype=float)

    A = np.array([
        [10.0,  3.0, 17.0,  3.5,  1.7,  8.0],
        [ 0.05,10.0, 17.0,  0.1,  8.0, 14.0],
        [ 3.0,  3.5,  1.7, 10.0, 17.0,  8.0],
        [17.0,  8.0,  0.05,10.0,  0.1, 14.0]
    ], dtype=float)

    P = 1e-4 * np.array([
        [1312, 1696, 5569,  124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091,  381]
    ], dtype=float)

    u = x  # domain already [0,1]^6

    # inner_k = sum_j A[k,j] * (u_j - P[k,j])^2
    diff = u[:, None, :] - P[None, :, :]            # (n, 4, 6)
    inner = np.sum(A[None, :, :] * diff**2, axis=2) # (n, 4)

    outer = np.sum(alpha[None, :] * np.exp(-inner), axis=1)  # (n,)
    f_native = -(2.58 + outer) / 1.94                       # (n,)

    if f_min is not None:
        return -(f_native - f_min)  # anchored: max = 0 at minimizer
    return -f_native         


def powell_4d(x, dim=None):
    """Powell (quartic) function — fixed 4D.
    Input:  x ∈ [0,1]^4  → rescale to u = 9*x - 4 ∈ [-4,5]^4
    Block (a,b,c,d):
        f = (a + 10b)^2 + 5(c - d)^2 + (b - 2c)^4 + 10(a - d)^4
    Returns: -f(u) so the maximum is 0 at the minimizer (u=0 → x=4/9).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n, d = x.shape
    if d != 4:
        raise ValueError(f"powell_4d expects 4 columns, got d={d}")
    if dim is not None and dim != 4:
        raise ValueError(f"'dim' must be 4 for powell_4d, got dim={dim}")

    # Rescale [0,1]^4 → [-4,5]^4
    u = 9.0 * x - 4.0
    a, b, c, dvec = u[:, 0], u[:, 1], u[:, 2], u[:, 3]

    value = (
        (a + 10.0 * b) ** 2
        + 5.0 * (c - dvec) ** 2
        + (b - 2.0 * c) ** 4
        + 10.0 * (a - dvec) ** 4
    )
    return -value  # shape (n,)


def shekel_4d(x, dim=None, m=10, b=None, C=None, f_min=None):
    """Shekel function (fixed 4D).
    Domain: u ∈ [0,10]^4 (we rescale from x ∈ [0,1]^4 via u = 10*x).
    Standard form (m = 10):
        f(u) = - Σ_{i=1}^m 1 / ( ||u - C_i||^2 + b_i )
    Defaults use the classic Shekel-10 parameters from SFU’s benchmark set.
    If f_min is provided, returns -(f - f_min) to anchor max=0 at the global minimizer.
    Otherwise returns -f (unanchored).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 4:
        raise ValueError("Shekel is fixed 4D; expected x with 4 columns.")

    # Defaults for the canonical Shekel-10
    if b is None:
        if m != 10:
            raise ValueError("Default b is only defined for m=10. Provide b for other m.")
        b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5], dtype=float)  # shape (10,)
    else:
        b = np.asarray(b, dtype=float)
        if b.shape != (m,):
            raise ValueError(f"b must have shape ({m},)")

    if C is None:
        if m != 10:
            raise ValueError("Default C is only defined for m=10. Provide C for other m.")
        C_mat = np.array([
            [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
            [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
        ], dtype=float)  # shape (4,10)
        C = C_mat.T  # shape (10,4); each row is a center C_i
    else:
        C = np.asarray(C, dtype=float)
        if C.shape != (m, 4):
            raise ValueError(f"C must have shape ({m}, 4)")

    # Rescale [0,1]^4 → [0,10]^4
    u = 10.0 * x  # (n, 4)

    # Compute denominators: ||u - C_i||^2 + b_i for all samples
    diff = u[:, None, :] - C[None, :, :]       # (n, m, 4)
    sqdist = np.sum(diff**2, axis=2)           # (n, m)
    terms = 1.0 / (sqdist + b[None, :])        # (n, m)
    f_native = -np.sum(terms, axis=1)          # (n,)

    if f_min is not None:
        return -(f_native - f_min)  # anchored: max = 0 at minimizer
    return -f_native                # unanchored


def styblinski_tang_nd(x, dim=None, f_min_1d=-39.16599):
    """Styblinski–Tang (nD).
    Native domain: u_j ∈ [-5, 5]
    f(u) = 0.5 * Σ_j (u_j^4 - 16 u_j^2 + 5 u_j)
    Global minimizer per-dim: u* ≈ -2.903534  →  f_min ≈ d * (-39.16599)
    Returns: -(f(u) - d*f_min_1d)  so the returned maximum is 0 at the minimizer.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n, d_infer = x.shape
    if dim is not None and dim != d_infer:
        raise ValueError(f"dim mismatch: dim={dim}, but x has {d_infer} columns")
    d = d_infer

    # Rescale [0,1]^d → [-5,5]^d
    u = 10.0 * x - 5.0

    f_native = 0.5 * np.sum(u**4 - 16.0*u**2 + 5.0*u, axis=1)

    # Anchor to 0 at the (approximate) global minimizer
    return -(f_native - d * f_min_1d)





FUNCTIONS_Batch3 = {
    'camel3': {
        'func': camel3_2d,
        'optimum': 0.0,
        'description': '2D only; polynomial valley with saddle; min at (0,0)',
        'dim': 2,
    },
    'camel6': {
        'func': camel6_2d,
        'optimum': 0.0,
        'description': '2D only; two global minima (±0.089842, ∓0.712656)',
        'dim': 2,
    },
    'dixon_price': {
        'func': dixon_price_nd,
        'optimum': 0.0,
        'description': 'nD; non-separable, curved valleys; closed-form minimizer',
        'dim': None,
    },
    'dejong5': {
        'func': dejong5_2d,
        'optimum': 0.0,
        'description': '2D only; Shekel’s Foxholes grid, many local minima',
        'dim': 2,
    },
    'easom': {
        'func': easom_2d,
        'optimum': 0.0,
        'description': '2D only; sharp global pit at (π,π); flat elsewhere',
        'dim': 2,
    },
    'michalewicz': {
        'func': michalewicz_nd,
        'optimum': None,  # unanchored unless f_min provided
        'description': 'nD; highly multimodal (controlled by m); unanchored by default',
        'dim': None,
    },
    'beale': {
        'func': beale_2d,
        'optimum': 0.0,
        'description': '2D only; polynomial with cross-terms; min at (3, 0.5)',
        'dim': 2,
    },
    'branin': {
        'func': branin_2d,
        'optimum': 0.0,
        'description': '2D only; 3 global minima (standard params)',
        'dim': 2,
    },
    'colville': {
        'func': colville_4d,
        'optimum': 0.0,
        'description': '4D only; Rosenbrock-like couplings; min at (1,1,1,1)',
        'dim': 4,
    },
    'forrester08': {
        'func': forretal08_1d,
        'optimum': 0.0,
        'description': '1D only; classic BO test function (nonstationary)',
        'dim': 1,
    },
    'goldstein_price': {
        'func': goldstein_price_2d,
        'optimum': 0.0,
        'description': '2D only; complex polynomial; min at (0,-1)',
        'dim': 2,
    },
    'hartmann3': {
        'func': hartmann3_3d,
        'optimum': None,  # unanchored unless f_min provided
        'description': '3D only; multimodal; native domain [0,1]^3',
        'dim': 3,
    },
    'hartmann4': {
        'func': hartmann4_4d,
        'optimum': None,  # unanchored unless f_min provided
        'description': '4D only; multimodal; native domain [0,1]^4',
        'dim': 4,
    },
    'hartmann6': {
        'func': hartmann6_6d,
        'optimum': None,  # unanchored unless f_min provided
        'description': '6D only; multimodal; native domain [0,1]^6',
        'dim': 6,
    },
    'powell': {
        'func': powell_4d,
        'optimum': 0.0,
        'description': 'nD; quartic blocks, requires d%4==0; min at u=0',
        'dim': 4,
    },
    'shekel': {
        'func': shekel_4d,
        'optimum': None,  # unanchored unless f_min provided (depends on m,b,C)
        'description': '4D only; Shekel m=10 by default; highly multimodal',
        'dim': 4,
    },
    'styblinski_tang': {
        'func': styblinski_tang_nd,
        'optimum': 0.0,
        'description': 'nD; nonconvex polynomial; per-dim min at u≈-2.903534',
        'dim': None,
    }
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
    unique_dims = sorted({cfg["dim"] for cfg in FUNCTIONS_Batch3.values() if cfg["dim"] is not None})
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
    for func_name in FUNCTIONS_Batch3.keys():
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
        sum(len(dims_for_func(func_info, DIMENSIONS)) for func_info in FUNCTIONS_Batch3.values())
        * len(ALPHA_CONFIGS)
        * len(G_CONFIGS)
        * len(noise_models)
        * len(noise_levels)
        * len(SEEDS)
    )


    print(f"Dimensions tested: {DIMENSIONS}")
    print(f"FUNCTIONS_Batch3: {list(FUNCTIONS_Batch3.keys())}")
    print(f"Noise models: {noise_models}")
    for noise_model in noise_models:
        for noise_std in noise_levels:
            print("  - " + noise_model_summary(noise_model, float(noise_std), noise_hyperparams))
    print(f"Total experiments: {total_experiments}")
    print()

    experiment_count = 0
    noise_slug = _noise_slug(noise_models)

    for func_name, func_info in FUNCTIONS_Batch3.items():
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

            with open(model_output_dir.joinpath(f"s03_batch3_{noise_slug}_raw_progress_ai_2.pkl"), "wb") as f:
                pickle.dump(results, f)


    df = analyze_results(results)
    df.to_csv(model_output_dir.joinpath(f"s03_batch3_{noise_slug}_experiment_results_ai_2.csv"), index=False)
    with open(model_output_dir.joinpath(f"s03_batch3_{noise_slug}_experiment_results_ai_2.pkl"), "wb") as f:
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
