// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! SIMD-accelerated implementation of Gumbel distribution functions.
//!
//! This module provides vectorised implementations of the Gumbel (Type I extreme value)
//! distribution's probability density function (PDF), cumulative distribution function (CDF),
//! and quantile function using SIMD instructions. The implementation automatically falls back
//! to scalar computation for unaligned data or when SIMD is not available.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::{Simd, StdFloat};

use minarrow::utils::is_simd_aligned;
use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_f64_simd_to, masked_univariate_kernel_f64_simd_to,
};
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std_to, masked_univariate_kernel_f64_std_to,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// SIMD-accelerated Gumbel PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// f(x; μ, β) = (1/β) exp(-z - exp(-z)) where z = (x - μ)/β
#[inline(always)]
pub fn gumbel_pdf_simd_to(
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

    const N: usize = W64;
    let inv_b = 1.0 / scale;
    let loc_v = Simd::<f64, N>::splat(location);
    let inv_b_v = Simd::<f64, N>::splat(inv_b);

    let scalar_body = |xi: f64| {
        let z = (xi - location) * inv_b;
        inv_b * (-(z + (-z).exp())).exp()
    };

    let simd_body = move |x_v: Simd<f64, N>| {
        let z = (x_v - loc_v) * inv_b_v;
        inv_b_v * (-(z + (-z).exp())).exp()
    };

    // Dense fast path
    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(x) {
            dense_univariate_kernel_f64_simd_to::<N, _, _>(x, output, simd_body, scalar_body);
        } else {
            dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        }
        return Ok(());
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("gumbel_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    if is_simd_aligned(x) {
        masked_univariate_kernel_f64_simd_to::<N, _, _>(
            x,
            mask_ref,
            output,
            &mut out_mask,
            simd_body,
            scalar_body,
        );
    } else {
        masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);
    }

    Ok(())
}

/// SIMD-accelerated implementation of Gumbel distribution probability density function.
///
/// Computes the probability density function (PDF) of the Gumbel (Type I extreme value)
/// distribution using vectorised SIMD operations for enhanced performance on large
/// datasets of extreme value analysis.
///
/// ## Parameters
/// - `x`: Input values where PDF should be evaluated (domain: all real numbers)
/// - `location`: Location parameter μ (shift parameter)
/// - `scale`: Scale parameter β > 0 (spread parameter)
/// - `null_mask`: Optional bitmask indicating null values in input
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing:
/// - **Success**: FloatArray with computed PDF values and appropriate null mask
/// - **Error**: KernelError::InvalidArguments for invalid location or scale parameters
///
/// ## Errors
/// - `KernelError::InvalidArguments`: When scale ≤ 0 or parameters are non-finite
///
/// ## Example Usage
/// ```rust,ignore
/// use minarrow::vec64;
/// let x = vec64![-1.0, 0.0, 1.0, 2.0];
/// let location = 0.0;  // μ = 0
/// let scale = 1.0;     // β = 1
/// let result = gumbel_pdf_simd(&x, location, scale, None, None)?;
/// // Returns PDF values for standard Gumbel distribution
/// ```
#[inline(always)]
pub fn gumbel_pdf_simd(
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

    gumbel_pdf_simd_to(
        x,
        location,
        scale,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// SIMD-accelerated Gumbel CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// F(x; μ, β) = exp(−exp(−z)), z = (x − μ)/β
#[inline(always)]
pub fn gumbel_cdf_simd_to(
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

    const N: usize = W64;
    let inv_b = 1.0 / scale;
    let loc_v = Simd::<f64, N>::splat(location);
    let inv_b_v = Simd::<f64, N>::splat(inv_b);

    let simd_body = move |x_v: Simd<f64, N>| {
        let z = (x_v - loc_v) * inv_b_v;
        (-(-z).exp()).exp()
    };

    let scalar_body = |xi: f64| {
        let z = (xi - location) * inv_b;
        (-(-z).exp()).exp()
    };

    // Dense fast path
    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(x) {
            dense_univariate_kernel_f64_simd_to::<N, _, _>(x, output, simd_body, scalar_body);
        } else {
            dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        }
        return Ok(());
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("gumbel_cdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    if is_simd_aligned(x) {
        masked_univariate_kernel_f64_simd_to::<N, _, _>(
            x,
            mask_ref,
            output,
            &mut out_mask,
            simd_body,
            scalar_body,
        );
    } else {
        masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);
    }

    Ok(())
}

/// SIMD-accelerated implementation of Gumbel distribution cumulative distribution function.
///
/// Computes the cumulative distribution function (CDF) of the Gumbel (Type I extreme value)
/// distribution using vectorised SIMD operations for enhanced performance on extreme
/// value statistical analysis.
///
/// ## Parameters
/// - `x`: Input values where CDF should be evaluated (domain: all real numbers)
/// - `location`: Location parameter μ (shift parameter)
/// - `scale`: Scale parameter β > 0 (spread parameter)
/// - `null_mask`: Optional bitmask indicating null values in input
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing:
/// - **Success**: FloatArray with computed CDF values and appropriate null mask
/// - **Error**: KernelError::InvalidArguments for invalid location or scale parameters
///
/// ## Errors
/// - `KernelError::InvalidArguments`: When scale ≤ 0 or parameters are non-finite
///
/// ## Use cases
/// The Gumbel CDF represents the probability of not exceeding a given threshold,
/// making it essential for:
/// - Risk assessment (probability of extreme events)
/// - Return period calculations (T = 1/(1-F(x)))
/// - Reliability analysis and safety margins
/// - Environmental impact studies
///
/// ## Example Usage
/// ```rust,ignore
/// let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
/// let location = 0.0;  // μ = 0
/// let scale = 1.0;     // β = 1
/// let result = gumbel_cdf_simd(&x, location, scale, None, None)?;
/// // Returns cumulative probabilities for standard Gumbel
/// ```
#[inline(always)]
pub fn gumbel_cdf_simd(
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

    gumbel_cdf_simd_to(
        x,
        location,
        scale,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// SIMD-accelerated Gumbel quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// Q(p; μ, β) = μ − β · ln(−ln(p))
#[inline(always)]
pub fn gumbel_quantile_simd_to(
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

    const N: usize = W64;
    let loc_v = Simd::<f64, N>::splat(location);
    let scale_v = Simd::<f64, N>::splat(scale);

    let scalar_body = |pi: f64| -> f64 { location - scale * (-pi.ln()).ln() };
    let simd_body = move |p_v: Simd<f64, N>| loc_v - scale_v * (-p_v.ln()).ln();

    // Dense fast path
    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(p) {
            dense_univariate_kernel_f64_simd_to::<N, _, _>(p, output, simd_body, scalar_body);
        } else {
            dense_univariate_kernel_f64_std_to(p, output, scalar_body);
        }
        return Ok(());
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("gumbel_quantile: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    if is_simd_aligned(p) {
        masked_univariate_kernel_f64_simd_to::<N, _, _>(
            p,
            mask_ref,
            output,
            &mut out_mask,
            simd_body,
            scalar_body,
        );
    } else {
        masked_univariate_kernel_f64_std_to(p, mask_ref, output, &mut out_mask, scalar_body);
    }

    Ok(())
}

/// SIMD-accelerated implementation of Gumbel distribution quantile function.
///
/// Computes the quantile function (inverse CDF) of the Gumbel (Type I extreme value)
/// distribution using vectorised SIMD operations for enhanced performance on
/// extreme value quantile calculations.
///
/// ## Parameters
/// - `p`: Input probability values where quantiles should be evaluated (domain: p ∈ (0, 1))
/// - `location`: Location parameter μ (shift parameter)
/// - `scale`: Scale parameter β > 0 (spread parameter)
/// - `null_mask`: Optional bitmask indicating null values in input
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing:
/// - **Success**: FloatArray with computed quantile values and appropriate null mask
/// - **Error**: KernelError::InvalidArguments for invalid location or scale parameters
///
/// ## Errors
/// - `KernelError::InvalidArguments`: When scale ≤ 0 or parameters are non-finite
///
/// ## Example Usage
/// ```rust,ignore
/// use minarrow::vec64;
/// let p = vec64![0.01, 0.1, 0.5, 0.9, 0.99];  // probability values
/// let location = 0.0;                    // μ = 0
/// let scale = 1.0;                       // β = 1
/// let result = gumbel_quantile_simd(&p, location, scale, None, None)?;
/// // Returns quantile values for standard Gumbel distribution
/// ```
#[inline(always)]
pub fn gumbel_quantile_simd(
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

    gumbel_quantile_simd_to(
        p,
        location,
        scale,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
