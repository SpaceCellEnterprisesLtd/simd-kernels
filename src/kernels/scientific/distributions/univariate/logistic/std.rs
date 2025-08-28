// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

use minarrow::{Bitmask, FloatArray, Vec64};
use std::f64::{INFINITY, NEG_INFINITY};

use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std_to, masked_univariate_kernel_f64_std_to,
};
use minarrow::enums::error::KernelError;

use crate::utils::has_nulls;

/// Logistic PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// f(x|μ, s) = exp(-(x-μ)/s) / [s * (1 + exp(-(x-μ)/s))²]
#[inline(always)]
pub fn logistic_pdf_std_to(
    x: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // Parameter checks
    if scale <= 0.0 || !scale.is_finite() || !location.is_finite() {
        return Err(KernelError::InvalidArguments(
            "logistic_pdf: invalid parameters".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let scalar_body = move |xi: f64| -> f64 {
        let z = (xi - location) / scale;
        let ez = (-z).exp();
        ez / (scale * (1.0 + ez).powi(2))
    };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("logistic_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);

    Ok(())
}

/// Logistic PDF (x|μ, s) = exp(-(x-μ)/s) / [s * (1 + exp(-(x-μ)/s))^2]
#[inline(always)]
pub fn logistic_pdf_std(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    logistic_pdf_std_to(
        x,
        location,
        scale,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Logistic CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// F(x|μ, s) = 1 / (1 + exp(-(x-μ)/s))
#[inline(always)]
pub fn logistic_cdf_std_to(
    x: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // Parameter checks
    if scale <= 0.0 || !scale.is_finite() || !location.is_finite() {
        return Err(KernelError::InvalidArguments(
            "logistic_cdf: invalid parameters".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let inv_s = 1.0 / scale;

    let scalar_body = move |xi: f64| -> f64 {
        let z = (xi - location) * inv_s;
        1.0 / (1.0 + (-z).exp())
    };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);

    Ok(())
}

/// Logistic CDF: F(x|μ, s) = 1 / (1 + exp(-(x-μ)/s))
#[inline(always)]
pub fn logistic_cdf_std(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    logistic_cdf_std_to(
        x,
        location,
        scale,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Logistic quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// Q(p) = location + scale * ln(p / (1 - p))
#[inline(always)]
pub fn logistic_quantile_std_to(
    p: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // Parameter checks
    if !location.is_finite() || !scale.is_finite() || scale <= 0.0 {
        return Err(KernelError::InvalidArguments(
            "logistic_quantile: invalid location or scale".into(),
        ));
    }
    if p.is_empty() {
        return Ok(());
    }

    let scalar_body = move |pi: f64| -> f64 {
        // Q(p | μ, s) = μ + s · ln( p / (1-p) ) only for 0 < p < 1  and  p finite
        if pi > 0.0 && pi < 1.0 && pi.is_finite() {
            location + scale * (pi.ln() - (1.0 - pi).ln())
        } else if pi == 0.0 {
            NEG_INFINITY
        } else if pi == 1.0 {
            INFINITY
        } else {
            f64::NAN
        }
    };

    // Dense fast path
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(p, output, scalar_body);
        return Ok(());
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(p, mask_ref, output, &mut out_mask, scalar_body);

    Ok(())
}

/// Logistic quantile (inverse CDF) function.
///
/// For a given probability `p` in (0,1), the quantile is:
///     Q(p) = location + scale * ln(p / (1 - p))
///
/// # Arguments
/// - `p`: Probabilities (must all be in (0,1)).
/// - `location`: Location parameter.
/// - `scale`: Scale parameter (must be positive).
///
/// # Returns
/// - `FloatArray<f64>` of quantiles.
/// - Returns error if any argument is invalid or out of range.
#[inline(always)]
pub fn logistic_quantile_std(
    p: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    logistic_quantile_std_to(
        p,
        location,
        scale,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
