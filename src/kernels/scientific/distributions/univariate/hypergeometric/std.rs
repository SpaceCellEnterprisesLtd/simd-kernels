// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::shared::scalar::*;
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_u64_std_to, masked_univariate_kernel_u64_std_to,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// Hypergeometric PMF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// P(X = k) = C(K,k) × C(N-K, n-k) / C(N,n)
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn hypergeometric_pmf_std_to(
    k: &[u64],
    population: u64,
    success: u64,
    draws: u64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // parameter checks
    if population == 0 || success > population || draws > population {
        return Err(KernelError::InvalidArguments(
            "hypergeometric_pmf: invalid parameters".into(),
        ));
    }
    if k.is_empty() {
        return Ok(());
    }

    // common pre-computations
    let ln_denom = ln_choose(population, draws);
    let min_k = success.min(draws);

    // scalar body (u64 -> f64)
    let scalar_body = |ki: u64| -> f64 {
        if ki <= min_k && draws >= ki && draws - ki <= population - success {
            (ln_choose(success, ki) + ln_choose(population - success, draws - ki) - ln_denom).exp()
        } else {
            0.0
        }
    };

    // choose dense vs. masked
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_u64_std_to(k, output, scalar_body);
        return Ok(());
    }

    // masked path
    let in_mask = null_mask.expect("hypergeometric_pmf: null_count > 0 requires null_mask");
    let mut out_mask = in_mask.clone();
    masked_univariate_kernel_u64_std_to(k, in_mask, output, &mut out_mask, scalar_body);

    Ok(())
}

/// Hypergeometric PMF: P(X = k)
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn hypergeometric_pmf_std(
    k: &[u64],
    population: u64,
    success: u64,
    draws: u64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = k.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    hypergeometric_pmf_std_to(
        k,
        population,
        success,
        draws,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Hypergeometric CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// F(k) = ∑_{i=0}^k PMF(i)
#[inline(always)]
pub fn hypergeometric_cdf_std_to(
    k: &[u64],
    population: u64,
    success: u64,
    draws: u64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if population == 0 || success > population || draws > population {
        return Err(KernelError::InvalidArguments(
            "hypergeometric_cdf: invalid parameters".into(),
        ));
    }
    if k.is_empty() {
        return Ok(());
    }

    let min_k = draws.min(success);
    let ln_denom = ln_choose(population, draws);

    if !has_nulls(null_count, null_mask) {
        for (idx, &ki) in k.iter().enumerate() {
            let kmax = ki.min(min_k);
            let mut sum = 0.0;
            for s in 0..=kmax {
                if draws >= s && draws - s <= population - success {
                    sum += (ln_choose(success, s) + ln_choose(population - success, draws - s)
                        - ln_denom)
                        .exp();
                }
            }
            output[idx] = sum;
        }
        return Ok(());
    }

    let mask = null_mask.expect("null_count > 0 requires null_mask");

    for idx in 0..k.len() {
        if !unsafe { mask.get_unchecked(idx) } {
            output[idx] = f64::NAN;
            continue;
        }
        let ki = k[idx];
        let kmax = ki.min(min_k);
        let mut sum = 0.0;
        for s in 0..=kmax {
            if draws >= s && draws - s <= population - success {
                sum += (ln_choose(success, s) + ln_choose(population - success, draws - s)
                    - ln_denom)
                    .exp();
            }
        }
        output[idx] = sum;
    }

    Ok(())
}

/// Hypergeometric CDF: F(k) = ∑_{i=0}^k PMF(i)
#[inline(always)]
pub fn hypergeometric_cdf_std(
    k: &[u64],
    population: u64,
    success: u64,
    draws: u64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = k.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    hypergeometric_cdf_std_to(
        k,
        population,
        success,
        draws,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Hypergeometric quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// Q(p) = smallest k such that CDF(k) ≥ p
#[inline(always)]
pub fn hypergeometric_quantile_std_to(
    p: &[f64],
    population: u64,
    success: u64,
    draws: u64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if population == 0 || success > population || draws > population {
        return Err(KernelError::InvalidArguments(
            "hypergeometric_quantile: invalid parameters".into(),
        ));
    }
    if p.is_empty() {
        return Ok(());
    }

    let n = draws;
    let kmax = draws.min(success);
    let ln_denom = ln_choose(population, draws);

    let compute_quantile = |pi: f64| -> f64 {
        // early exit
        // 0.0 returns -1, 1.0 returns kmax, and outside of that is NaN
        if !pi.is_finite() || pi < 0.0 || pi > 1.0 + 1e-14 {
            return f64::NAN;
        }
        if pi == 0.0 {
            return -1.0;
        }
        if pi >= 1.0 - 1e-14 {
            // Handle values very close to 1.0
            return kmax as f64;
        }

        // main logic
        let mut cdf = 0.0;
        let mut qk = kmax; // default to kmax if no k satisfies condition
        for k in 0..=kmax {
            if n >= k && n - k <= population - success {
                let pmf = (ln_choose(success, k) + ln_choose(population - success, n - k)
                    - ln_denom)
                    .exp();
                cdf += pmf;
                if cdf >= pi {
                    qk = k;
                    break;
                }
            }
        }
        qk as f64
    };

    if !has_nulls(null_count, null_mask) {
        for (idx, &pi) in p.iter().enumerate() {
            output[idx] = compute_quantile(pi);
        }
        return Ok(());
    }

    let mask = null_mask.expect("null_count > 0 requires null_mask");

    for idx in 0..p.len() {
        if !mask.get(idx) {
            output[idx] = f64::NAN;
        } else {
            output[idx] = compute_quantile(p[idx]);
        }
    }

    Ok(())
}

/// Hypergeometric quantile: Q(p) = smallest k such that CDF(k) ≥ p
#[inline(always)]
pub fn hypergeometric_quantile_std(
    p: &[f64],
    population: u64,
    success: u64,
    draws: u64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    hypergeometric_quantile_std_to(
        p,
        population,
        success,
        draws,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
