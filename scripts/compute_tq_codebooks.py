#!/usr/bin/env python3
"""
Compute optimal Lloyd-Max codebooks for TurboQuant quantization.

The codebooks are optimal scalar quantizers for the marginal distribution of each
coordinate of a uniformly random point on the unit sphere S^{d-1} (Lemma 1 from
Zandieh et al., ICLR 2026). The PDF is:

    f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

which in high dimensions converges to N(0, 1/d).

The Lloyd-Max algorithm iteratively refines centroids by:
    1. Set boundaries as midpoints between adjacent centroids
    2. Update each centroid to the conditional mean E[X | boundary_lo <= X < boundary_hi]
    3. Repeat until convergence

Usage:
    python scripts/compute_tq_codebooks.py [--dims 64 128] [--bits 3 4]
"""

import argparse
import logging

import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
from scipy.stats import norm

logger = logging.getLogger(__name__)


def beta_pdf(x, d):
    """PDF of each coordinate of a uniform random point on S^{d-1}."""
    if abs(x) >= 1:
        return 0.0
    coeff = gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))
    return coeff * (1 - x**2) ** ((d - 3) / 2)


def lloyd_max(d, n_levels, n_iter=200):
    """Compute optimal Lloyd-Max codebook for the coordinate distribution at dimension d."""
    sigma = 1.0 / np.sqrt(d)
    centroids = np.array(
        [norm.ppf((i + 0.5) / n_levels, scale=sigma) for i in range(n_levels)]
    )

    for iteration in range(n_iter):
        boundaries = (
            [-1.0]
            + [(centroids[i] + centroids[i + 1]) / 2 for i in range(n_levels - 1)]
            + [1.0]
        )

        new_centroids = []
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            num, _ = quad(lambda x: x * beta_pdf(x, d), lo, hi)
            den, _ = quad(lambda x: beta_pdf(x, d), lo, hi)
            if den > 1e-30:
                new_centroids.append(num / den)
            else:
                new_centroids.append(centroids[i])
        centroids = np.array(new_centroids)

    return centroids


def compute_mse(d, centroids):
    """Compute the expected MSE per coordinate for the given codebook."""
    n_levels = len(centroids)
    boundaries = (
        [-1.0]
        + [(centroids[i] + centroids[i + 1]) / 2 for i in range(n_levels - 1)]
        + [1.0]
    )
    mse = 0.0
    for i in range(n_levels):
        lo, hi = boundaries[i], boundaries[i + 1]
        val, _ = quad(lambda x: (x - centroids[i]) ** 2 * beta_pdf(x, d), lo, hi)
        mse += val
    return mse


def format_c_array(name, centroids):
    """Format centroids as a C static array."""
    lines = [f"static const float {name}[{len(centroids)}] = {{"]
    for i in range(0, len(centroids), 2):
        pair = centroids[i : i + 2]
        entries = ", ".join(f"{v: .17e}f" for v in pair)
        comma = "," if i + 2 < len(centroids) else ""
        lines.append(f"    {entries}{comma}")
    lines.append("};")
    return "\n".join(lines)


def boundaries_of(centroids):
    """Decision boundaries are midpoints between adjacent centroids (mirrors
    ggml-quants.c:tq_compute_boundaries). Outputs n-1 values for n centroids.
    """
    return [(centroids[i] + centroids[i + 1]) / 2 for i in range(len(centroids) - 1)]


def format_glsl_array(name, values):
    """Format values as a GLSL const float array (for copy_to_quant.comp)."""
    lines = [f"const float {name}[{len(values)}] = float[{len(values)}]("]
    for i, v in enumerate(values):
        comma = "," if i + 1 < len(values) else ""
        lines.append(f"    {v: .17f}{comma}")
    lines.append(");")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute TurboQuant Lloyd-Max codebooks")
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 128],
                        help="Head dimensions to compute codebooks for")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4],
                        help="Bit-widths to compute codebooks for")
    parser.add_argument("--c-code", action="store_true",
                        help="Output C code ready to paste into ggml-quants.c")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for d in args.dims:
        logger.info("\n%s", "=" * 60)
        logger.info("  d = %d  (sigma ≈ %.6f)", d, 1 / np.sqrt(d))
        logger.info("%s", "=" * 60)

        for b in args.bits:
            n_levels = 1 << b
            centroids = lloyd_max(d, n_levels)
            mse = compute_mse(d, centroids)

            logger.info("\n  %d-bit (%d centroids), MSE/coord = %.8f", b, n_levels, mse)
            logger.info("  Total MSE (d coords) = %.8f", d * mse)

            if args.c_code:
                cb_name = f"TQ{b}_CODEBOOK_{d}"
                logger.info("\n%s", format_c_array(cb_name, centroids))
                # Also emit the GLSL decision boundaries (midpoints between
                # adjacent centroids), used by the Vulkan encoder
                # copy_to_quant.comp to pick indices. These MUST track the
                # codebook for FA / mul_mat to reproduce the CPU-reference
                # quantization on the GPU.
                bnd = boundaries_of(centroids)
                glsl_name = f"TBQ{b}_B"
                logger.info("\n// d=%d decision boundaries (midpoints of %s).", d, cb_name)
                logger.info("// Gate this with #if defined(TQ_D64) in copy_to_quant.comp for d=64.")
                logger.info("%s", format_glsl_array(glsl_name, bnd))
            else:
                for i, c in enumerate(centroids):
                    logger.info("    [%2d] % .17f", i, c)
                logger.info("    boundaries (midpoints):")
                for i, v in enumerate(boundaries_of(centroids)):
                    logger.info("      [%2d] % .17f", i, v)


if __name__ == "__main__":
    main()
