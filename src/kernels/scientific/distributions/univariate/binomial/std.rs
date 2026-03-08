// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Binomial Distribution Scalar Implementation**
//!
//! Scalar implementations of binomial distribution functions using
//! traditional mathematical algorithms optimised for accuracy and numerical stability.
//!
//! ## Overview
//! This module provides the scalar (non-SIMD) reference implementations for binomial
//! distribution calculations. These implementations serve as both fallback implementations
//! when SIMD is unavailable.

use minarrow::enums::error::KernelError;
use minarrow::{Bitmask, FloatArray, Vec64, vec64};

use crate::kernels::scientific::distributions::shared::scalar::*;
use crate::{
    kernels::scientific::distributions::univariate::common::std::{
        dense_univariate_kernel_f64_std_to, masked_univariate_kernel_f64_std_to,
    },
    utils::has_nulls,
};

/// Binomial PMF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn binomial_pmf_std_to(
    k: &[u64],
    n: u64,
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if !(0.0 <= p && p <= 1.0) || !p.is_finite() {
        return Err(KernelError::InvalidArguments(
            "binomial_pmf: invalid p".into(),
        ));
    }
    if k.is_empty() {
        return Ok(());
    }

    let ln_choose_n = ln_gamma((n as f64) + 1.0);

    let scalar_body = |ki_f: f64| -> f64 {
        let ki = ki_f as u64;
        if ki > n {
            return 0.0;
        }
        if p == 0.0 {
            return if ki == 0 { 1.0 } else { 0.0 };
        }
        if p == 1.0 {
            return if ki == n { 1.0 } else { 0.0 };
        }
        (ln_choose_n - ln_gamma((ki as f64) + 1.0) - ln_gamma(((n - ki) as f64) + 1.0)
            + (ki as f64) * p.ln()
            + ((n - ki) as f64) * (1.0 - p).ln())
        .exp()
    };

    let k_f64: Vec<f64> = k.iter().map(|&ki| ki as f64).collect();

    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(&k_f64, output, scalar_body);
        return Ok(());
    }

    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let mut out_mask = mask.clone();
    masked_univariate_kernel_f64_std_to(&k_f64, mask, output, &mut out_mask, scalar_body);
    Ok(())
}

/// Scalar implementation of binomial distribution probability mass function.
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn binomial_pmf_std(
    k: &[u64],
    n: u64,
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

    binomial_pmf_std_to(k, n, p, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Binomial CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn binomial_cdf_std_to(
    k: &[u64],
    n: u64,
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if !(p >= 0.0 && p <= 1.0) || n > 1_000_000_000 {
        return Err(KernelError::InvalidArguments(
            "binomial_cdf: invalid p or n".into(),
        ));
    }
    if k.is_empty() {
        return Ok(());
    }

    let n_f = n as f64;

    let scalar_body = |ki_f: f64| -> f64 {
        let ki = ki_f as u64;
        if ki >= n {
            1.0
        } else if p == 0.0 {
            1.0
        } else if p == 1.0 {
            if ki >= n { 1.0 } else { 0.0 }
        } else {
            1.0 - incomplete_beta((ki + 1) as f64, n_f - ki as f64, p)
        }
    };

    let k_f64: Vec<f64> = k.iter().map(|&ki| ki as f64).collect();

    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(&k_f64, output, scalar_body);
        return Ok(());
    }

    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let mut out_mask = mask.clone();
    masked_univariate_kernel_f64_std_to(&k_f64, mask, output, &mut out_mask, scalar_body);
    Ok(())
}

/// Scalar implementation of binomial distribution cumulative distribution function.
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn binomial_cdf_std(
    k: &[u64],
    n: u64,
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

    binomial_cdf_std_to(k, n, p, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Binomial quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn binomial_quantile_std_to(
    p: &[f64],
    n: u64,
    p_: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if !(0.0 < p_ && p_ < 1.0) || !p_.is_finite() {
        return Err(KernelError::InvalidArguments(
            "binomial_quantile: invalid p_".into(),
        ));
    }
    if n == 0 {
        for i in 0..p.len() {
            output[i] = 0.0;
        }
        return Ok(());
    }
    if p.is_empty() {
        return Ok(());
    }

    let small_n = n <= 100;
    let small_cdf: Option<Vec<f64>> = if small_n {
        let mut cdf = vec![0.0; (n + 1) as usize];
        let mut prob = (1.0 - p_).powf(n as f64);
        cdf[0] = prob;
        for k in 1..=n as usize {
            prob *= ((n - k as u64 + 1) as f64) * p_ / (k as f64 * (1.0 - p_));
            cdf[k] = cdf[k - 1] + prob;
        }
        Some(cdf)
    } else {
        None
    };

    let scalar_body = |pi: f64| -> f64 {
        if !(pi >= 0.0 && pi <= 1.0) || !pi.is_finite() {
            return f64::NAN;
        }
        if pi == 0.0 {
            return -1.0;
        }
        if pi == 1.0 {
            return n as f64;
        }
        if small_n {
            let cdf = small_cdf.as_ref().unwrap();
            let mut k = 0_usize;
            while k < cdf.len() && cdf[k] < pi {
                k += 1;
            }
            k as f64
        } else {
            binomial_quantile_cornish_fisher(pi, n, p_)
        }
    };

    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(p, output, scalar_body);
        return Ok(());
    }

    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let mut out_mask = mask.clone();
    masked_univariate_kernel_f64_std_to(p, mask, output, &mut out_mask, scalar_body);
    Ok(())
}

/// Scalar implementation of binomial distribution quantile function (inverse CDF).
#[inline(always)]
pub fn binomial_quantile_std(
    p: &[f64],
    n: u64,
    p_: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = p.len();
    if n == 0 {
        return Ok(FloatArray::from_vec64(vec64![0.0; len], None));
    }
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    binomial_quantile_std_to(p, n, p_, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
