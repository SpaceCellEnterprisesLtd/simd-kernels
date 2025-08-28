//! Standard (scalar) implementation of geometric distribution functions.
//!
//! This module provides scalar implementations of the geometric distribution's probability
//! mass function (PMF), cumulative distribution function (CDF), and quantile function
//! (inverse CDF). These implementations serve as the fallback when SIMD is not available
//! and are optimised for scalar computation patterns.
//!
//! ## Implementation Strategy
//!
//! ### SciPy Convention Adherence
//! All functions follow SciPy's "number of trials" convention:
//! - Support on {1, 2, 3, ...} with P(X = 0) = 0 and F(0) = 0
//! - Consistent with `scipy.stats.geom` behaviour
//!
//! ### Numerical Optimisations
//! - **PMF**: Direct computation for moderate k, log-space for large k
//! - **CDF**: Optimised power computation with special cases for p ≈ 1
//! - **Quantile**: Efficient logarithmic inversion with integer ceiling
//!
//! ### Robustness Features
//! - Comprehensive parameter validation
//! - Graceful handling of boundary cases (k = 0, p = 1)
//! - Automatic precision management for extreme parameter values

// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::{
    kernels::scientific::distributions::univariate::common::std::{
        dense_univariate_kernel_f64_std, dense_univariate_kernel_f64_std_to,
        masked_univariate_kernel_f64_std, masked_univariate_kernel_f64_std_to,
    },
    utils::has_nulls,
};
use minarrow::enums::error::KernelError;

/// Geometric PMF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn geometric_pmf_std_to(
    k: &[u64],
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if !(p > 0.0 && p <= 1.0) || !p.is_finite() {
        return Err(KernelError::InvalidArguments(
            "geometric_pmf: p must be in (0,1] and finite".into(),
        ));
    }
    if k.is_empty() {
        return Ok(());
    }

    // Degenerate p == 1 -> pmf(k) = 1_{k==1}
    if p == 1.0 {
        if !has_nulls(null_count, null_mask) {
            for (i, &ki) in k.iter().enumerate() {
                output[i] = if ki == 1 { 1.0 } else { 0.0 };
            }
            return Ok(());
        }
        let mask = null_mask.expect("null_count > 0 requires null_mask");
        for i in 0..k.len() {
            if !mask.get(i) {
                output[i] = f64::NAN;
            } else {
                output[i] = if k[i] == 1 { 1.0 } else { 0.0 };
            }
        }
        return Ok(());
    }

    // 0 < p < 1
    let log1mp = (1.0 - p).ln();
    let scalar_body = |ki: u64| -> f64 {
        if ki == 0 {
            0.0
        } else {
            (((ki as f64) - 1.0) * log1mp).exp() * p
        }
    };

    if !has_nulls(null_count, null_mask) {
        for (i, &ki) in k.iter().enumerate() {
            output[i] = scalar_body(ki);
        }
        return Ok(());
    }

    let mask = null_mask.expect("null_count > 0 requires null_mask");
    for i in 0..k.len() {
        if !mask.get(i) {
            output[i] = f64::NAN;
        } else {
            output[i] = scalar_body(k[i]);
        }
    }
    Ok(())
}

/// Geometric PMF (SciPy convention):
/// P(X = k) = (1 - p)^(k-1) · p for k = 1,2,…  and P(0) = 0.
/// Accepts p ∈ (0,1] with p=1 being the degenerate-at-1 distribution.
/// Returns Err if p ∉ [0,1] or non-finite.
#[inline(always)]
pub fn geometric_pmf_std(
    k: &[u64],
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = k.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    geometric_pmf_std_to(k, p, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Geometric CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn geometric_cdf_std_to(
    k: &[u64],
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if !(p > 0.0 && p <= 1.0) || !p.is_finite() {
        return Err(KernelError::InvalidArguments(
            "geometric_cdf: p must be in (0,1] and finite".into(),
        ));
    }
    if k.is_empty() {
        return Ok(());
    }

    // Degenerate p == 1 -> CDF(k) = 0 if k==0 else 1
    if p == 1.0 {
        if !has_nulls(null_count, null_mask) {
            for (i, &ki) in k.iter().enumerate() {
                output[i] = if ki == 0 { 0.0 } else { 1.0 };
            }
            return Ok(());
        }
        let mask = null_mask.expect("null_count > 0 requires null_mask");
        for i in 0..k.len() {
            if !mask.get(i) {
                output[i] = f64::NAN;
            } else {
                output[i] = if k[i] == 0 { 0.0 } else { 1.0 };
            }
        }
        return Ok(());
    }

    // 0 < p < 1
    let log1mp = (1.0 - p).ln();

    if !has_nulls(null_count, null_mask) {
        for (i, &ki) in k.iter().enumerate() {
            output[i] = if ki == 0 {
                0.0
            } else {
                1.0 - ((ki as f64) * log1mp).exp()
            };
        }
        return Ok(());
    }

    let mask = null_mask.expect("null_count > 0 requires null_mask");
    for i in 0..k.len() {
        if !mask.get(i) {
            output[i] = f64::NAN;
        } else {
            let ki = k[i];
            output[i] = if ki == 0 {
                0.0
            } else {
                1.0 - ((ki as f64) * log1mp).exp()
            };
        }
    }
    Ok(())
}

/// Geometric CDF (SciPy convention):
/// F(X ≤ k) = 0 for k = 0; and F(X ≤ k) = 1 - (1-p)^k for k ≥ 1.
/// Accepts p ∈ (0,1] with p=1 being the degenerate-at-1 distribution.
/// Returns Err if p ∉ [0,1] or non-finite.
#[inline(always)]
pub fn geometric_cdf_std(
    k: &[u64],
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = k.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    geometric_cdf_std_to(k, p, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Geometric quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn geometric_quantile_std_to(
    pv: &[f64],
    p_succ: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // parameter check
    if !(p_succ > 0.0 && p_succ < 1.0) || !p_succ.is_finite() {
        return Err(KernelError::InvalidArguments(
            "geometric_quantile: p_succ must be in (0,1) and finite".into(),
        ));
    }
    if pv.is_empty() {
        return Ok(());
    }

    let log1mp = (1.0 - p_succ).ln();

    let scalar_body = |q: f64| -> f64 {
        if !q.is_finite() || q < 0.0 || q > 1.0 {
            return f64::NAN;
        }
        if q == 0.0 {
            return 1.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        ((1.0 - q).ln() / log1mp).ceil()
    };

    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(pv, output, scalar_body);
        return Ok(());
    }

    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let mut out_mask = mask.clone();
    masked_univariate_kernel_f64_std_to(pv, mask, output, &mut out_mask, scalar_body);
    Ok(())
}

/// Geometric quantile: Q(p) = ceil(ln(1-p) / ln(1-p_succ)), for p ∈ (0,1) (scipy convention)
#[inline(always)]
pub fn geometric_quantile_std(
    pv: &[f64],
    p_succ: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = pv.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    geometric_quantile_std_to(pv, p_succ, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
