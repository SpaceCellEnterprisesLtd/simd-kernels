// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Beta Distribution Scalar Implementation**
//!
//! High-precision scalar implementations of beta distribution functions using
//! traditional mathematical algorithms optimised for accuracy and numerical stability.
//!
//! ## Overview
//! This module provides the scalar (non-SIMD) reference implementations for beta
//! distribution calculations. These implementations prioritise numerical accuracy
//! and serve as both fallback implementations when SIMD is unavailable and as
//! reference implementations for validation.
//!
//! ## Implementation Details
//! - **PDF**: Uses logarithmic computation to avoid overflow, with special handling
//!   for boundary cases (x=0, x=1) based on parameter values
//! - **CDF**: Employs continued fraction expansion of the regularised incomplete beta
//!   function I_x(α, β) for maximum accuracy
//! - **Quantile**: Implements AS 109 algorithm with Newton-Raphson refinement
//!
//! ## Numerical Stability
//! - Logarithmic scaling prevents overflow in PDF calculations
//! - Continued fractions provide numerically stable CDF computation
//! - Special case handling for boundary conditions and parameter edge cases
//! - Consistent with SciPy reference implementations

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::shared::scalar::*;
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::dense_univariate_kernel_f64_std_to;
use crate::kernels::scientific::distributions::univariate::common::std::masked_univariate_kernel_f64_std_to;
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// Scalar implementation of beta distribution PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn beta_pdf_std_to(
    x: &[f64],
    alpha: f64,
    beta: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if alpha <= 0.0 || beta <= 0.0 || !alpha.is_finite() || !beta.is_finite() {
        return Err(KernelError::InvalidArguments(
            "beta_pdf: invalid alpha or beta".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let log_norm = -ln_gamma(alpha) - ln_gamma(beta) + ln_gamma(alpha + beta);

    let scalar_body = |xi: f64| -> f64 {
        if xi < 0.0 || xi > 1.0 {
            0.0
        } else if xi == 0.0 || xi == 1.0 {
            // Handle boundary cases
            if (alpha < 1.0 && xi == 0.0) || (beta < 1.0 && xi == 1.0) {
                f64::INFINITY
            } else if (alpha > 1.0 && xi == 0.0) || (beta > 1.0 && xi == 1.0) {
                0.0
            } else if alpha == 1.0 && xi == 0.0 {
                beta
            } else if beta == 1.0 && xi == 1.0 {
                alpha
            } else {
                // alpha == 1.0 && beta == 1.0, uniform distribution
                1.0
            }
        } else {
            ((alpha - 1.0) * xi.ln() + (beta - 1.0) * (1.0 - xi).ln() + log_norm).exp()
        }
    };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    // Null-aware masked path
    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let mut out_mask = mask.clone();
    masked_univariate_kernel_f64_std_to(x, mask, output, &mut out_mask, scalar_body);
    Ok(())
}

/// Scalar implementation of beta distribution probability density function.
///
/// Computes the PDF using logarithmic computation for numerical stability:
/// f(x) = exp((α-1)ln(x) + (β-1)ln(1-x) + ln_norm)
/// where ln_norm = ln(Γ(α+β)) - ln(Γ(α)) - ln(Γ(β))
///
/// ## Accuracy
/// Relative error typically < 1e-14 compared to mathematical definition.
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn beta_pdf_std(
    x: &[f64],
    alpha: f64,
    beta: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    beta_pdf_std_to(x, alpha, beta, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Scalar implementation of beta distribution CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn beta_cdf_std_to(
    x: &[f64],
    alpha: f64,
    beta: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if alpha <= 0.0 || beta <= 0.0 || !alpha.is_finite() || !beta.is_finite() {
        return Err(KernelError::InvalidArguments(
            "beta_cdf: invalid alpha or beta".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let scalar_body = |xi: f64| -> f64 {
        if xi <= 0.0 {
            0.0
        } else if xi >= 1.0 {
            1.0
        } else {
            incomplete_beta(alpha, beta, xi)
        }
    };

    if !has_nulls(null_count, null_mask) {
        for (i, &xi) in x.iter().enumerate() {
            output[i] = scalar_body(xi);
        }
        return Ok(());
    }

    // Null-aware masked path
    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let mut out_mask = mask.clone();
    masked_univariate_kernel_f64_std_to(x, mask, output, &mut out_mask, scalar_body);

    Ok(())
}

/// Scalar implementation of beta distribution cumulative distribution function.
///
/// Computes the CDF using the regularised incomplete beta function I_x(α, β)
/// via continued fraction expansion for maximum numerical stability and accuracy.
///
/// ## Accuracy
/// Relative error typically < 1e-14, consistent with SciPy implementations.
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn beta_cdf_std(
    x: &[f64],
    alpha: f64,
    beta: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    beta_cdf_std_to(x, alpha, beta, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Scalar implementation of beta distribution quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn beta_quantile_std_to(
    p: &[f64],
    alpha: f64,
    beta: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if alpha <= 0.0 || beta <= 0.0 || !alpha.is_finite() || !beta.is_finite() {
        return Err(KernelError::InvalidArguments(
            "beta_quantile: invalid alpha or beta".into(),
        ));
    }
    if p.is_empty() {
        return Ok(());
    }

    let scalar_body = |pi: f64| -> f64 {
        if !(pi >= 0.0 && pi <= 1.0) {
            f64::NAN
        } else if pi == 0.0 {
            0.0
        } else if pi == 1.0 {
            1.0
        } else {
            incomplete_beta_inv(alpha, beta, pi)
        }
    };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        for (i, &pi) in p.iter().enumerate() {
            output[i] = scalar_body(pi);
        }
        return Ok(());
    }

    // Null-aware masked path
    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let mut out_mask = mask.clone();
    masked_univariate_kernel_f64_std_to(p, mask, output, &mut out_mask, scalar_body);

    Ok(())
}

/// Scalar implementation of beta distribution quantile function (inverse CDF).
///
/// Computes the quantile function using the AS 109 algorithm with Newton-Raphson
/// refinement to solve F(x) = p for x, where F is the beta CDF.
#[inline(always)]
pub fn beta_quantile_std(
    p: &[f64],
    alpha: f64,
    beta: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    beta_quantile_std_to(p, alpha, beta, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
