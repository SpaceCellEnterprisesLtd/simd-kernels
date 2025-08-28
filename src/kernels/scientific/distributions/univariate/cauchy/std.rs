//! # **Cauchy Distribution Scalar Implementation**
//!
//! Scalar implementations using standard library transcendental functions.
//! PDF uses efficient rational function, CDF/quantile use atan/tan functions.

// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

use minarrow::{Bitmask, FloatArray, Vec64};
use std::f64::consts::PI;

use crate::kernels::scientific::distributions::shared::constants::*;
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std, dense_univariate_kernel_f64_std_to,
    masked_univariate_kernel_f64_std, masked_univariate_kernel_f64_std_to,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// Cauchy PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// f(x; location, scale) = (1/π)·[scale / ((x − location)² + scale²)]
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn cauchy_pdf_std_to(
    x: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // Parameter checks
    if !(scale > 0.0 && scale.is_finite()) || !location.is_finite() {
        return Err(KernelError::InvalidArguments(
            "cauchy_pdf: invalid location or scale".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let inv_scale = 1.0 / scale;
    let inv_ps = INV_PI * inv_scale;

    // scalar fallback
    let scalar_body = move |xi: f64| {
        let z = (xi - location) * inv_scale;
        inv_ps / (1.0 + z * z)
    };

    // Dense path - no nulls
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    // Null-aware path
    let mask = null_mask.expect("cauchy_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask.clone();
    masked_univariate_kernel_f64_std_to(x, mask, output, &mut out_mask, scalar_body);

    Ok(())
}

/// Cauchy PDF (vectorised, SIMD where available), null-aware and Arrow-compliant.
/// f(x; location, scale) = (1/π)·[scale / ((x − location)² + scale²)]
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn cauchy_pdf_std(
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

    cauchy_pdf_std_to(
        x,
        location,
        scale,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Cauchy CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// F(x; location, scale) = ½ + (1/π)·atan((x − location)/scale)
#[inline(always)]
pub fn cauchy_cdf_std_to(
    x: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // Validate parameters
    if !(scale > 0.0 && scale.is_finite()) || !location.is_finite() {
        return Err(KernelError::InvalidArguments(
            "cauchy_cdf: invalid location or scale".into(),
        ));
    }
    let len = x.len();
    if len == 0 {
        return Ok(());
    }
    let inv_scale = 1.0 / scale;

    // Fast dense path - no nulls
    if !has_nulls(null_count, null_mask) {
        for (i, &xi) in x.iter().enumerate() {
            let z = (xi - location) * inv_scale;
            output[i] = 0.5 + INV_PI * z.atan();
        }
        return Ok(());
    }

    // Null-aware path
    let mask = null_mask.expect("cauchy_cdf: null_count > 0 requires null_mask");

    for idx in 0..len {
        if !mask.get(idx) {
            output[idx] = f64::NAN;
        } else {
            let xi = x[idx];
            let z = (xi - location) * inv_scale;
            output[idx] = 0.5 + INV_PI * z.atan();
        }
    }

    Ok(())
}

/// Cauchy CDF (scalar), null-aware and Arrow-compliant.
/// F(x; location, scale) = ½ + (1/π)·atan((x − location)/scale)
#[inline(always)]
pub fn cauchy_cdf_std(
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

    cauchy_cdf_std_to(
        x,
        location,
        scale,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Cauchy quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// Q(p; location, scale) = location + scale · tan[π·(p - ½)]
#[inline(always)]
pub fn cauchy_quantile_std_to(
    p: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // Validate parameters
    if !(scale > 0.0 && scale.is_finite()) || !location.is_finite() {
        return Err(KernelError::InvalidArguments(
            "cauchy_quantile: invalid location or scale".into(),
        ));
    }
    let len = p.len();
    if len == 0 {
        return Ok(());
    }

    // Fast dense path - no nulls
    if !has_nulls(null_count, null_mask) {
        for (i, &pi) in p.iter().enumerate() {
            if pi == 0.0 {
                output[i] = f64::NEG_INFINITY;
            } else if pi == 1.0 {
                output[i] = f64::INFINITY;
            } else if pi > 0.0 && pi < 1.0 {
                // Robust tails
                let t = pi.min(1.0 - pi);
                if t <= 1e-8 {
                    // asymptotic: tan(π(p-1/2)) = -cot(πp) ~ -1/(πp) for p->0
                    let s = if pi < 0.5 { -1.0 } else { 1.0 };
                    let tail = s / (PI * t);
                    output[i] = location + scale * tail;
                } else {
                    // interior: use FMA to reduce cancellation
                    let angle = pi.mul_add(PI, -0.5 * PI);
                    output[i] = location + scale * angle.tan();
                }
            } else {
                output[i] = f64::NAN;
            }
        }
        return Ok(());
    }

    // Null-aware path
    let mask = null_mask.expect("cauchy_quantile: null_count > 0 requires null_mask");

    for idx in 0..len {
        if !unsafe { mask.get_unchecked(idx) } {
            // preserve input null
            output[idx] = f64::NAN;
        } else {
            let pi = p[idx];

            // honour -inf/inf at the endpoints; outside (0,1) -> NaN
            if pi == 0.0 {
                output[idx] = f64::NEG_INFINITY;
            } else if pi == 1.0 {
                output[idx] = f64::INFINITY;
            } else if pi > 0.0 && pi < 1.0 {
                // robust tails
                let t = pi.min(1.0 - pi);
                if t <= 1e-8 {
                    // tan(π(p−½)) = −cot(πp) ≈ −1/(πp) for p->0
                    let s = if pi < 0.5 { -1.0 } else { 1.0 };
                    let tail = s / (PI * t);
                    output[idx] = location + scale * tail;
                } else {
                    // interior: fused multiply-add for better precision
                    let angle = pi.mul_add(PI, -0.5 * PI);
                    output[idx] = location + scale * angle.tan();
                }
            } else {
                // produce NaN but mark valid
                output[idx] = f64::NAN;
            }
        }
    }

    Ok(())
}

/// Cauchy quantile (inverse CDF): Q(p; location, scale) = location + scale · tan[π·(p - ½)]
#[inline(always)]
pub fn cauchy_quantile_std(
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

    cauchy_quantile_std_to(
        p,
        location,
        scale,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
