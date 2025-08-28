// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Weibull Distribution Scalar Implementations** - *Reliability and Survival Analysis Foundation*
//!
//! Scalar implementations of Weibull distribution functions optimised for reliability engineering
//! and survival analysis applications. These implementations emphasise numerical stability across
//! the wide parameter ranges encountered in practical applications.

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, dense_univariate_kernel_f64_std_to,
    masked_univariate_kernel_f64_std, masked_univariate_kernel_f64_std_to,
};
use minarrow::enums::error::KernelError;

use crate::utils::has_nulls;

/// Weibull PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// f(x|k,λ) = (k/λ) × (x/λ)^(k-1) × exp(-(x/λ)^k) for x ≥ 0, else 0
#[inline(always)]
pub fn weibull_pdf_std_to(
    x: &[f64],
    shape: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // parameter checks
    if !(shape > 0.0 && shape.is_finite()) || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "weibull_pdf: invalid shape or scale".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let inv_scale = 1.0 / scale;
    let coeff = shape * inv_scale; // k/λ

    // scalar kernel
    let scalar_body = move |xi: f64| -> f64 {
        if xi < 0.0 {
            0.0
        } else {
            let t = xi * inv_scale; // x/λ
            let t_pow_k = t.powf(shape); // (x/λ)^k
            coeff * t.powf(shape - 1.0) * (-t_pow_k).exp()
        }
    };

    // dense (null-free) path
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    // null-aware path
    let mask_ref = null_mask.expect("weibull_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);

    Ok(())
}

#[inline(always)]
pub fn weibull_pdf_std(
    x: &[f64],
    shape: f64,
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

    weibull_pdf_std_to(x, shape, scale, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Weibull CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// F(x|k,λ) = 1 − exp[−(x/λ)^k] for x ≥ 0, else 0
#[inline(always)]
pub fn weibull_cdf_std_to(
    x: &[f64],
    shape: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // parameter checks
    if !(shape > 0.0 && shape.is_finite()) || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "weibull_cdf: invalid shape or scale".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let inv_scale = 1.0 / scale;

    let scalar_body = move |xi: f64| -> f64 {
        if xi < 0.0 {
            0.0
        } else {
            1.0 - (-(xi * inv_scale).powf(shape)).exp()
        }
    };

    // dense (null-free) path
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    // null-aware path
    let mask_ref = null_mask.expect("weibull_cdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);

    Ok(())
}

/// Weibull CDF: F(x; k, λ) = 1 − exp[−(x/λ)^k]  for x ≥ 0, else 0
#[inline(always)]
pub fn weibull_cdf_std(
    x: &[f64],
    shape: f64,
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

    weibull_cdf_std_to(x, shape, scale, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Weibull quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// Q(p|k,λ) = λ · [−ln(1−p)]^(1/k), p ∈ (0,1)
#[inline(always)]
pub fn weibull_quantile_std_to(
    p: &[f64],
    shape: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // parameter checks
    if !(shape > 0.0 && shape.is_finite()) || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "weibull_quantile: invalid shape or scale".into(),
        ));
    }
    if p.is_empty() {
        return Ok(());
    }

    let inv_k = 1.0 / shape;

    let scalar_body = move |pi: f64| -> f64 {
        // 0.0 returns 0 , 1.0 returns inf
        // outside these ranges returns NaN
        if 0.0 < pi && pi < 1.0 {
            scale * (-(1.0 - pi).ln()).powf(inv_k)
        } else if pi == 0.0 {
            0.0
        } else if pi == 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    };

    // dense (null-free) path
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(p, output, scalar_body);
        return Ok(());
    }

    // null-aware path
    let mask_ref = null_mask.expect("weibull_quantile: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(p, mask_ref, output, &mut out_mask, scalar_body);

    Ok(())
}

/// Weibull quantile (inverse CDF):
/// Q(p; k, λ) = λ · [−ln(1−p)]^(1/k),  p ∈ (0,1)
#[inline(always)]
pub fn weibull_quantile_std(
    p: &[f64],
    shape: f64,
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

    weibull_quantile_std_to(p, shape, scale, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
