// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Uniform Distribution Scalar Implementations** - *Perfect Simplicity and Precision*
//!
//! Scalar implementations of uniform distribution functions.

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std_to, masked_univariate_kernel_f64_std_to,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// Uniform PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn uniform_pdf_std_to(
    x: &[f64],
    a: f64,
    b: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if !(a < b) || !a.is_finite() || !b.is_finite() {
        return Err(KernelError::InvalidArguments(
            "uniform_pdf: a must be < b and finite".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let inv_width = 1.0 / (b - a);

    let scalar_body = move |xi: f64| -> f64 {
        if xi.is_nan() {
            f64::NAN
        } else if xi >= a && xi <= b {
            inv_width
        } else {
            0.0
        }
    };

    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("uniform_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);
    Ok(())
}

/// Uniform PDF: f(x|a,b) = 1/(b-a) for x in [a, b], 0 otherwise.
#[inline(always)]
pub fn uniform_pdf_std(
    x: &[f64],
    a: f64,
    b: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    uniform_pdf_std_to(x, a, b, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Uniform CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn uniform_cdf_std_to(
    x: &[f64],
    a: f64,
    b: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if !(a < b) || !a.is_finite() || !b.is_finite() {
        return Err(KernelError::InvalidArguments(
            "uniform_cdf: a must be < b and both finite".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let inv_width = 1.0 / (b - a);

    let scalar_body = |xi: f64| -> f64 {
        if xi.is_nan() {
            f64::NAN
        } else if xi < a {
            0.0
        } else if xi > b {
            1.0
        } else {
            (xi - a) * inv_width
        }
    };

    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("uniform_cdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);
    Ok(())
}

/// Uniform CDF: F(x|a,b) = 0 if x < a, (x-a)/(b-a) if x in [a,b], 1 if x > b.
#[inline(always)]
pub fn uniform_cdf_std(
    x: &[f64],
    a: f64,
    b: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    uniform_cdf_std_to(x, a, b, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Uniform quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn uniform_quantile_std_to(
    p: &[f64],
    a: f64,
    b: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if !(a < b) || !a.is_finite() || !b.is_finite() {
        return Err(KernelError::InvalidArguments(
            "uniform_quantile: a must be < b and finite".into(),
        ));
    }
    if p.is_empty() {
        return Ok(());
    }

    let width = b - a;

    let scalar_body = move |pi: f64| -> f64 {
        if (0.0..=1.0).contains(&pi) && pi.is_finite() {
            a + pi * width
        } else {
            f64::NAN
        }
    };

    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(p, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("uniform_quantile: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(p, mask_ref, output, &mut out_mask, scalar_body);
    Ok(())
}

/// Continuous-uniform quantile
#[inline(always)]
pub fn uniform_quantile_std(
    p: &[f64],
    a: f64,
    b: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    uniform_quantile_std_to(p, a, b, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
