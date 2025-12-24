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



def create_experiment_config(dimensions=[2, 5, 10]):
    """Create experiment configuration for different dimensions"""
    
    configs = {}
    unique_dims = sorted({cfg['dim'] for cfg in FUNCTIONS_Batch3.values() if cfg['dim'] is not None})
    all_dims = dimensions + unique_dims
    for dim in all_dims:
        # Scale parameters with dimension
        n_init = max(2*dim, 5)
        n_iter = min(10*dim, 200) #max(20, 5 * dim)  # Scale iterations with dimension
        n_iter = max(n_iter, 30) # at least 30 iterations for small problems
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
    
    for func_name in FUNCTIONS_Batch3.keys():
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
    total_experiments = (len(FUNCTIONS_Batch3) * len(DIMENSIONS) * len(ALPHA_CONFIGS) * 
                        len(G_CONFIGS) * len(NOISE_LEVELS) * len(SEEDS))
    
    print(f"Dimensions tested: {DIMENSIONS}")
    print(f"FUNCTIONS_Batch3: {list(FUNCTIONS_Batch3.keys())}")
    print(f"Total experiments (upper bound): {total_experiments}")
    print()
    
    experiment_count = 0


    for func_name, func_info in FUNCTIONS_Batch3.items():
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
            with open(model_output_dir.joinpath('s03_batch3_raw_experiment_results.pkl'), 'wb') as f:
                pickle.dump(results, f)
    df = analyze_results(results)
    df.to_csv(model_output_dir.joinpath('s03_batch3_df_experiment_results.csv'))
    
    return results

if __name__ == "__main__":
    # data setup
    arguments = parse_args()
    print(arguments["model_output_dir"])
    results = run_benchmark(model_output_dir = arguments["model_output_dir"])
   



    
    

