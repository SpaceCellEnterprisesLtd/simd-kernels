// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! Standard (scalar) implementation of Gumbel distribution functions.
//!
//! This module provides scalar implementations of the Gumbel (Type I extreme value)
//! distribution's probability density function (PDF), cumulative distribution function (CDF),
//! and quantile function. These implementations serve as the fallback when SIMD is not
//! available and are optimised for scalar computation patterns.

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std_to, masked_univariate_kernel_f64_std_to,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// Gumbel PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// f(x; μ, β) = (1/β) exp(-z - exp(-z)) where z = (x - μ)/β
#[inline(always)]
pub fn gumbel_pdf_std_to(
    x: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // Parameter checks
    if !location.is_finite() || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "gumbel_pdf: invalid location or scale".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let inv_b = 1.0 / scale;

    let scalar_body = |xi: f64| {
        let z = (xi - location) * inv_b;
        inv_b * (-(z + (-z).exp())).exp()
    };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("gumbel_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);

    Ok(())
}

/// Gumbel (type I extreme value)
#[inline(always)]
pub fn gumbel_pdf_std(
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

    gumbel_pdf_std_to(
        x,
        location,
        scale,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Gumbel CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// F(x; μ, β) = exp(−exp(−z)),   z = (x − μ)/β
#[inline(always)]
pub fn gumbel_cdf_std_to(
    x: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // Parameter checks
    if !location.is_finite() || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "gumbel_cdf: invalid location or scale".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let inv_b = 1.0 / scale;

    let scalar_body = |xi: f64| {
        let z = (xi - location) * inv_b;
        (-(-z).exp()).exp()
    };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("gumbel_cdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);

    Ok(())
}

/// CDF of the Gumbel (type I extreme) distribution:
/// F(x; μ, β) = exp(−exp(−z)),   z = (x − μ)/β
#[inline(always)]
pub fn gumbel_cdf_std(
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

    gumbel_cdf_std_to(
        x,
        location,
        scale,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Gumbel quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// Q(p; μ, β) = μ − β · ln(−ln(p))
#[inline(always)]
pub fn gumbel_quantile_std_to(
    p: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // Parameter validation
    if !location.is_finite() || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "gumbel_quantile: invalid location or scale".into(),
        ));
    }
    if p.is_empty() {
        return Ok(());
    }

    let scalar_body = |pi: f64| -> f64 { location - scale * (-pi.ln()).ln() };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(p, output, scalar_body);
        return Ok(());
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("gumbel_quantile: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(p, mask_ref, output, &mut out_mask, scalar_body);

    Ok(())
}

/// Gumbel quantile (inverse CDF), null-aware and SIMD-accelerated.
/// Q(p; μ, β) = μ − β · ln(−ln(p))
#[inline(always)]
pub fn gumbel_quantile_std(
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

    gumbel_quantile_std_to(
        p,
        location,
        scale,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
