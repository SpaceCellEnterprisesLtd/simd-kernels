// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

use minarrow::{Bitmask, FloatArray, Vec64};

#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::shared::constants::*;
use crate::kernels::scientific::distributions::shared::scalar::*;
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::erf::erf;
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, dense_univariate_kernel_f64_std_to,
    masked_univariate_kernel_f64_std, masked_univariate_kernel_f64_std_to,
};

/// Lognormal PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// f(x|μ,σ) = 1/(xσ√2π) * exp(-½[(ln(x)-μ)/σ]^2)
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn lognormal_pdf_std_to(
    x: &[f64],
    meanlog: f64,
    sdlog: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    use crate::kernels::scientific::distributions::shared::constants::SQRT_2PI;

    // Parameter checks
    if sdlog <= 0.0 || !sdlog.is_finite() || !meanlog.is_finite() {
        return Err(KernelError::InvalidArguments(
            "lognormal_pdf: invalid parameters".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let denom = sdlog * SQRT_2PI;

    let scalar_body = |xi: f64| -> f64 {
        if xi > 0.0 {
            let z = (xi.ln() - meanlog) / sdlog;
            (-0.5 * z * z).exp() / (denom * xi)
        } else {
            0.0
        }
    };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    // Null-aware path
    let mask_ref = null_mask.expect("lognormal_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);
    Ok(())
}

/// Lognormal PDF: f(x|μ,σ) = 1/(xσ√2π) * exp(-½[(ln(x)-μ)/σ]^2)
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn lognormal_pdf_std(
    x: &[f64],
    meanlog: f64,
    sdlog: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    lognormal_pdf_std_to(x, meanlog, sdlog, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Lognormal CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// F(x|μ,σ) = Φ((ln(x) - μ)/σ)
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn lognormal_cdf_std_to(
    x: &[f64],
    meanlog: f64,
    sdlog: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    use crate::kernels::scientific::distributions::shared::constants::SQRT_2;

    // Parameter checks
    if sdlog <= 0.0 || !sdlog.is_finite() || !meanlog.is_finite() {
        return Err(KernelError::InvalidArguments(
            "lognormal_cdf: invalid parameters".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    // Constants
    let sqrt2 = SQRT_2;

    // scalar fallback
    let scalar_body = move |xi: f64| -> f64 {
        if xi > 0.0 {
            let z = (xi.ln() - meanlog) / sdlog / sqrt2;
            0.5 * (1.0 + erf(z))
        } else {
            0.0
        }
    };

    // Dense fast path
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    // Null‐aware path
    let mask_ref = null_mask.expect("lognormal_cdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);

    Ok(())
}

/// Lognormal CDF: F(x|μ,σ) = Φ((ln(x) - μ)/σ)
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn lognormal_cdf_std(
    x: &[f64],
    meanlog: f64,
    sdlog: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    lognormal_cdf_std_to(x, meanlog, sdlog, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Lognormal quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// Q(p|μ,σ) = exp(μ + σ * Φ⁻¹(p)), p in (0,1)
#[inline(always)]
pub fn lognormal_quantile_std_to(
    p: &[f64],
    meanlog: f64,
    sdlog: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if sdlog <= 0.0 || !sdlog.is_finite() || !meanlog.is_finite() {
        return Err(KernelError::InvalidArguments(
            "lognormal_quantile: invalid parameters".into(),
        ));
    }
    let len = p.len();
    if len == 0 {
        return Ok(());
    }

    // Dense fast path: no nulls
    if !has_nulls(null_count, null_mask) {
        for (i, &pi) in p.iter().enumerate() {
            let nq = normal_quantile_scalar(pi, 0.0, 1.0);
            output[i] = (meanlog + sdlog * nq).exp();
        }
        return Ok(());
    }

    // Propagate input mask nulls as is; retain any new `NaN` or `inf`
    // values verbatim
    let mask = null_mask.expect("null_count > 0 requires null_mask");

    for idx in 0..len {
        if !unsafe { mask.get_unchecked(idx) } {
            output[idx] = f64::NAN;
        } else {
            let pi = p[idx];
            let nq = normal_quantile_scalar(pi, 0.0, 1.0);
            output[idx] = (meanlog + sdlog * nq).exp();
        }
    }
    Ok(())
}

/// Lognormal quantile: Q(p|μ,σ) = exp(μ + σ * Φ⁻¹(p)), p in (0,1)
#[inline(always)]
pub fn lognormal_quantile_std(
    p: &[f64],
    meanlog: f64,
    sdlog: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    lognormal_quantile_std_to(p, meanlog, sdlog, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
