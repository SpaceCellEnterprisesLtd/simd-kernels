// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Negative Binomial Scalar Implementations** - *CPU-Optimised Discrete Statistics*
//!
//! Scalar (non-SIMD) implementations of negative binomial distribution functions optimised for
//! numerical stability and computational efficiency. These implementations serve as the foundation
//! for SIMD acceleration and provide reliable fallback behaviour.

use std::f64;

use crate::kernels::scientific::distributions::shared::scalar::*;
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_u64_std, dense_univariate_kernel_u64_std_to,
    masked_univariate_kernel_u64_std, masked_univariate_kernel_u64_std_to,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;
use minarrow::{Bitmask, FloatArray, Vec64};

/// Negative Binomial PMF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn neg_binomial_pmf_std_to(
    k: &[u64],
    r: u64,
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // 1) Parameter checks
    if r == 0 || !(p > 0.0 && p < 1.0) || !p.is_finite() {
        return Err(KernelError::InvalidArguments(
            "neg_binomial_pmf: r must be positive, p ∈ (0,1), and both finite".into(),
        ));
    }
    // 2) Empty input
    if k.is_empty() {
        return Ok(());
    }

    let log_p = p.ln();
    let log1mp = (1.0 - p).ln();
    let r_f64 = r as f64;

    let scalar_body = move |ki: u64| {
        let lf = ln_choose(ki + r - 1, ki);
        (lf + r_f64 * log_p + (ki as f64) * log1mp).exp()
    };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_u64_std_to(k, output, scalar_body);
        return Ok(());
    }

    // Null‐aware masked path
    let mask_ref = null_mask.expect("neg_binomial_pmf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_u64_std_to(k, mask_ref, output, &mut out_mask, scalar_body);

    Ok(())
}

/// Negative Binomial PMF (Pascal distribution, number of failures before r-th success)
/// PMF: P(X=k) = C(k+r-1, k) * p^r * (1-p)^k, for k=0,1,...
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn neg_binomial_pmf_std(
    k: &[u64],
    r: u64,
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

    neg_binomial_pmf_std_to(k, r, p, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Negative Binomial CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn neg_binomial_cdf_std_to(
    k: &[u64],
    r: u64,
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if r == 0 || !(p > 0.0 && p < 1.0) || !p.is_finite() {
        return Err(KernelError::InvalidArguments(
            "neg_binomial_cdf: r must be positive, p ∈ (0,1), and both finite".into(),
        ));
    }
    let r_f64 = r as f64;
    let len = k.len();
    if len == 0 {
        return Ok(());
    }

    // Dense fast path
    if !has_nulls(null_count, null_mask) {
        for (i, &ki) in k.iter().enumerate() {
            output[i] = incomplete_beta(r_f64, (ki as f64) + 1.0, p);
        }
        return Ok(());
    }

    // Null-aware path
    let mask = null_mask.expect("null_count > 0 requires null_mask");

    for idx in 0..len {
        if !mask.get(idx) {
            output[idx] = f64::NAN;
        } else {
            let ki = k[idx];
            output[idx] = incomplete_beta(r_f64, (ki as f64) + 1.0, p);
        }
    }
    Ok(())
}

/// Negative Binomial CDF (sum of PMFs to k): F(X ≤ k) = I_p(r, k+1)
#[inline(always)]
pub fn neg_binomial_cdf_std(
    k: &[u64],
    r: u64,
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

    neg_binomial_cdf_std_to(k, r, p, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Negative Binomial Quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn neg_binomial_quantile_std_to(
    q: &[f64],
    r: u64,
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if r == 0 || !(p > 0.0 && p < 1.0) || !p.is_finite() {
        return Err(KernelError::InvalidArguments(
            "neg_binomial_quantile: r must be positive, p ∈ (0,1), and both finite".into(),
        ));
    }
    let r_f64 = r as f64;
    let len = q.len();
    if len == 0 {
        return Ok(());
    }

    // Quantile per probability entry
    let compute_quantile = |qi: f64| -> f64 {
        if qi > 0.0 && qi < 1.0 {
            let x = incomplete_beta_inv(r_f64, 1.0, qi);
            let mut k = x.floor().max(0.0) as u64;
            while incomplete_beta(r_f64, (k as f64) + 1.0, p) < qi {
                k += 1;
            }
            k as f64

        // 0 becomes -1, 1 becomes infinite
        // outside of these is NaN
        } else if qi == 0.0 {
            -1.0
        } else if qi == 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    };

    // Dense path
    if !has_nulls(null_count, null_mask) {
        for (i, &qi) in q.iter().enumerate() {
            output[i] = compute_quantile(qi);
        }
        return Ok(());
    }

    // Null-aware path
    let mask = null_mask.expect("null_count > 0 requires null_mask");

    for idx in 0..len {
        if !mask.get(idx) {
            output[idx] = f64::NAN;
        } else {
            output[idx] = compute_quantile(q[idx]);
        }
    }
    Ok(())
}

/// Negative Binomial Quantile (inverse CDF): returns minimal k such that CDF(k) >= q
#[inline(always)]
pub fn neg_binomial_quantile_std(
    q: &[f64],
    r: u64,
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = q.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    neg_binomial_quantile_std_to(q, r, p, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
