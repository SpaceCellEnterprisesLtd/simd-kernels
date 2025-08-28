// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Cauchy Distribution SIMD Implementation**
//!
//! High-performance SIMD-accelerated implementations of Cauchy distribution functions
//! utilising vectorised rational function computations for bulk PDF evaluations.
//!
//! ## Overview
//! This module provides SIMD (Single Instruction, Multiple Data) implementations
//! for the Cauchy (Lorentz) distribution PDF that process multiple values simultaneously
//! using CPU vector instructions. CDF and quantile functions remain scalar-only due
//! to the complexity of vectorising inverse tangent computations.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::Simd;

use minarrow::utils::is_simd_aligned;
use minarrow::{Bitmask, FloatArray};

use minarrow::Vec64;

use crate::kernels::scientific::distributions::shared::constants::*;
use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_f64_simd_to, masked_univariate_kernel_f64_simd_to,
};
use minarrow::enums::error::KernelError;

use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std_to, masked_univariate_kernel_f64_std_to,
};
use crate::utils::has_nulls;

/// SIMD-accelerated Cauchy PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// f(x; μ, σ) = (1/π) · σ / ((x - μ)² + σ²)
#[inline(always)]
pub fn cauchy_pdf_simd_to(
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

    const N: usize = W64;
    let inv_scale = 1.0 / scale;
    let inv_ps = INV_PI * inv_scale;

    // SIMD constants
    let loc_v = Simd::<f64, N>::splat(location);
    let inv_s_v = Simd::<f64, N>::splat(inv_scale);
    let inv_ps_v = Simd::<f64, N>::splat(inv_ps);

    // scalar fallback body
    let scalar_body = move |xi: f64| {
        let z = (xi - location) * inv_scale;
        inv_ps / (1.0 + z * z)
    };

    // SIMD body
    let simd_body = move |x_v: Simd<f64, N>| {
        let z = (x_v - loc_v) * inv_s_v;
        inv_ps_v / (Simd::splat(1.0) + z * z)
    };

    // Dense path (no nulls)
    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(x) {
            dense_univariate_kernel_f64_simd_to::<N, _, _>(x, output, simd_body, scalar_body);
        } else {
            dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        }
        return Ok(());
    }

    // Null-aware path
    let mask = null_mask.expect("cauchy_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask.clone();
    if is_simd_aligned(x) {
        masked_univariate_kernel_f64_simd_to::<N, _, _>(
            x,
            mask,
            output,
            &mut out_mask,
            simd_body,
            scalar_body,
        );
    } else {
        masked_univariate_kernel_f64_std_to(x, mask, output, &mut out_mask, scalar_body);
    }

    Ok(())
}

/// SIMD-accelerated implementation of Cauchy distribution probability density function.
///
/// Processes multiple PDF evaluations simultaneously using vectorised rational function
/// computations for great performance on large datasets with 64-byte memory alignment.
///
/// ## Mathematical Definition
/// ```text
/// f(x; μ, σ) = (1/π) · σ / ((x - μ)² + σ²)
/// ```
/// where μ is the location parameter and σ is the scale parameter.
///
/// ## Parameters
/// - `x`: Input values to evaluate (requires 64-byte alignment for SIMD path)
/// - `location`: Location parameter μ (must be finite)
/// - `scale`: Scale parameter σ (must be positive and finite)
/// - `null_mask`: Optional bitmask for null value handling
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing PDF values, or error for invalid parameters.
#[inline(always)]
pub fn cauchy_pdf_simd(
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

    cauchy_pdf_simd_to(
        x,
        location,
        scale,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
