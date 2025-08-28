// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Negative Binomial SIMD Implementations** - *Vectorised Discrete Distribution Computing*
//!
//! High-performance SIMD-accelerated implementations of negative binomial distribution functions
//! leveraging modern CPU vector instruction sets for optimal throughput on large vectors.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::f64;

use std::simd::{Simd, StdFloat, num::SimdUint};

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::shared::scalar::*;
use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_u64_simd_to, masked_univariate_kernel_u64_simd_to,
};
use minarrow::enums::error::KernelError;

use crate::utils::has_nulls;

/// Negative Binomial PMF SIMD (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn neg_binomial_pmf_simd_to(
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

    const N: usize = W64;
    let log_p = p.ln();
    let log1mp = (1.0 - p).ln();
    let r_f64 = r as f64;

    let scalar_body = move |ki: u64| {
        let lf = ln_choose(ki + r - 1, ki);
        (lf + r_f64 * log_p + (ki as f64) * log1mp).exp()
    };

    let simd_body = move |k_u: Simd<u64, N>| {
        let kf = k_u.cast::<f64>();
        // ln C(k+r-1, k) + r*ln(p) + k*ln(1-p)
        let ln_comb = ln_choose_v(kf + Simd::splat(r_f64) - Simd::splat(1.0), kf);
        let logpmf = ln_comb + Simd::splat(r_f64) * Simd::splat(log_p) + kf * Simd::splat(log1mp);
        logpmf.exp()
    };

    // Dense fast path (no nulls)
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_u64_simd_to::<N, _, _>(k, output, simd_body, scalar_body);
        return Ok(());
    }

    // Null‐aware masked path
    let mask_ref = null_mask.expect("neg_binomial_pmf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_u64_simd_to::<N, _, _>(
        k,
        mask_ref,
        output,
        &mut out_mask,
        simd_body,
        scalar_body,
    );

    Ok(())
}

/// **Negative Binomial Distribution Probability Mass Function** - *SIMD-Accelerated Discrete PMF*
///
/// Computes the probability mass function of the negative binomial distribution using vectorised
/// SIMD operations where possible, with automatic scalar fallback for optimal performance in
/// discrete probability computation and success/failure modelling.
///
/// ## Mathematical Definition
///
/// The negative binomial (Pascal) distribution PMF is defined as:
///
/// ```text
/// P(X=k|r,p) = C(k+r-1, k) × p^r × (1-p)^k
/// ```
///
/// Where:
/// - `X = k` ∈ ℕ₀: number of failures before r-th success (input values)
/// - `r` > 0: number of successes required (integer parameter)
/// - `p` ∈ (0,1): success probability on each trial
/// - `C(n,k)`: binomial coefficient "n choose k"
///
/// ## Parameters
///
/// * `k` - Input data slice of `u64` failure counts where PMF is evaluated
/// * `r` - Number of successes required, must be positive integer
/// * `p` - Success probability per trial, must be in (0,1) and finite
/// * `null_mask` - Optional input null bitmap for handling missing values
/// * `null_count` - Optional count of null values, enables optimised processing paths
///
/// ## Returns
///
/// Returns `Result<FloatArray<f64>, KernelError>` containing:
/// * **Success**: `FloatArray` with PMF values and appropriate null mask
/// * **Error**: `KernelError::InvalidArguments` for invalid parameters
///
/// ## Edge Cases & Error Handling
///
/// - **Invalid parameters**: Returns error if `r = 0`, `p ∉ (0,1)`, or `!p.is_finite()`
/// - **Null propagation**: Input nulls are preserved in output with corresponding mask bits
/// - **Empty input**: Returns empty `FloatArray` for zero-length input slices
/// - **Large counts**: Maintains numerical stability for extreme k values
/// - **Boundary probabilities**: Handles p very close to 0 or 1 with stability
#[inline(always)]
pub fn neg_binomial_pmf_simd(
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

    neg_binomial_pmf_simd_to(k, r, p, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
