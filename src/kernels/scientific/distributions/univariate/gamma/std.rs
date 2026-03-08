// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! Standard (scalar) implementation of gamma distribution functions.
//!
//! This module provides scalar implementations of the gamma distribution's probability
//! density function (PDF), cumulative distribution function (CDF), and quantile function
//! (inverse CDF). These implementations serve as the fallback when SIMD is not available
//! or for data that is not properly aligned for vectorised computation.
//!
//! ## Implementation Notes
//!
//! ### Rate Parameterisation
//! All functions use the rate parameterisation where β = 1/θ.
//!
//! ### Numerical Stability
//! - **PDF**: Uses log-space arithmetic to prevent overflow for large shape parameters
//! - **CDF**: Employs continued fraction expansion of the incomplete gamma function
//! - **Quantile**: Uses region-specific inversion algorithms for optimal accuracy
//!
//! ### Special Case Handling
//! - Zero and negative inputs are handled explicitly
//! - Boundary cases (p=0, p=1) return appropriate limit values
//! - Non-finite inputs and invalid parameters produce NaN or errors as appropriate

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::shared::scalar::*;
use minarrow::enums::error::KernelError;

#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, dense_univariate_kernel_f64_std_to,
    masked_univariate_kernel_f64_std, masked_univariate_kernel_f64_std_to,
};
use crate::utils::has_nulls;

/// Scalar implementation of gamma distribution PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn gamma_pdf_std_to(
    x: &[f64],
    shape: f64,
    rate: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if shape <= 0.0 || !shape.is_finite() || rate <= 0.0 || !rate.is_finite() {
        return Err(KernelError::InvalidArguments(
            "gamma_pdf: invalid shape or rate".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let beta = rate;
    let ln_gamma_k = ln_gamma(shape);
    let log_norm = shape * beta.ln() - ln_gamma_k;

    let scalar_body = |xi: f64| -> f64 {
        if !xi.is_finite() {
            return f64::NAN;
        }
        if xi < 0.0 {
            return 0.0;
        }
        if xi == 0.0 {
            return if shape > 1.0 {
                0.0
            } else if shape < 1.0 {
                f64::INFINITY
            } else {
                beta
            };
        }
        (log_norm + (shape - 1.0) * xi.ln() - beta * xi).exp()
    };

    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("gamma_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);
    Ok(())
}

#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn gamma_pdf_std(
    x: &[f64],
    shape: f64,
    rate: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    gamma_pdf_std_to(x, shape, rate, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Scalar implementation of gamma distribution CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn gamma_cdf_std_to(
    x: &[f64],
    shape: f64,
    rate: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if shape <= 0.0 || !shape.is_finite() || rate <= 0.0 || !rate.is_finite() {
        return Err(KernelError::InvalidArguments(
            "gamma_cdf: invalid shape or rate".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }
    let beta = rate;
    let len = x.len();

    let eval = |xi: f64| -> f64 {
        if !xi.is_finite() {
            return f64::NAN;
        }
        if xi <= 0.0 {
            return 0.0;
        }
        reg_lower_gamma(shape, beta * xi)
    };

    if !has_nulls(null_count, null_mask) {
        for (i, &xi) in x.iter().enumerate() {
            output[i] = eval(xi);
        }
        return Ok(());
    }

    let mask = null_mask.expect("Null mask must be present if null_count > 0");
    for i in 0..len {
        if !unsafe { mask.get_unchecked(i) } {
            output[i] = f64::NAN;
        } else {
            output[i] = eval(x[i]);
        }
    }
    Ok(())
}

/// Gamma CDF: F(x) = γ(k, x/θ) / Γ(k)
/// Gamma CDF with rate β: F(x) = P(k, β x) = γ(k, β x)/Γ(k)
#[inline(always)]
pub fn gamma_cdf_std(
    x: &[f64],
    shape: f64,
    rate: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    gamma_cdf_std_to(x, shape, rate, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Scalar implementation of gamma distribution quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn gamma_quantile_std_to(
    p: &[f64],
    shape: f64, // k > 0
    rate: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if shape <= 0.0 || !shape.is_finite() || rate <= 0.0 || !rate.is_finite() {
        return Err(KernelError::InvalidArguments(
            "gamma_quantile: invalid shape or rate".into(),
        ));
    }
    if p.is_empty() {
        return Ok(());
    }

    let beta = rate;
    let len = p.len();

    // Use upper-tail inversion when (1-p) is tiny (large-x regime).
    // Central & left regions go through the lower-tail inverter.
    const RIGHT_TAIL_SWITCH: f64 = 1e-10;

    let eval = |prob: f64| -> f64 {
        if !prob.is_finite() || prob < 0.0 || prob > 1.0 {
            return f64::NAN;
        }
        if prob == 0.0 {
            return 0.0;
        }
        if prob == 1.0 {
            return f64::INFINITY;
        }

        let q = 1.0 - prob;
        let z = if q <= RIGHT_TAIL_SWITCH {
            // robust for extreme right tail
            inv_reg_upper_gamma(shape, q)
        } else {
            // central / left
            inv_reg_lower_gamma(shape, prob)
        };
        z / beta
    };

    if !has_nulls(null_count, null_mask) {
        for (i, &pi) in p.iter().enumerate() {
            output[i] = eval(pi);
        }
        return Ok(());
    }

    let mask = null_mask.expect("Null mask must be present if null_count > 0");
    for i in 0..len {
        if !unsafe { mask.get_unchecked(i) } {
            output[i] = f64::NAN;
        } else {
            output[i] = eval(p[i]);
        }
    }
    Ok(())
}

/// Gamma quantile (inverse CDF): x such that CDF(x) = p
///
/// Gamma quantile (inverse CDF): x such that CDF(x) = p
#[inline(always)]
pub fn gamma_quantile_std(
    p: &[f64],
    shape: f64, // k > 0
    rate: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    gamma_quantile_std_to(p, shape, rate, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
