// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # Hypergeometric Distribution
//!
//! The hypergeometric distribution models the number of successes in a fixed number of draws
//! from a finite population without replacement. It is the discrete analogue of sampling
//! without replacement, contrasting with the binomial distribution which assumes sampling
//! with replacement.
//!
//! ## Mathematical Definition
//!
//! The hypergeometric distribution is parameterised by three non-negative integers:
//! - N: Population size
//! - K: Number of success items in the population  
//! - n: Number of draws (sample size)
//!
//! The probability mass function is:
//! ```text
//! P(X = k) = C(K, k) × C(N-K, n-k) / C(N, n)
//! ```
//!
//! Where C(a, b) = a! / (b!(a-b)!) is the binomial coefficient.
//!
//! ## Common Applications
//!
//! - **Quality control**: Defective items in batch sampling
//! - **Ecology**: Capture-recapture population studies
//! - **Card games**: Probability of specific hands without replacement
//! - **Survey sampling**: Stratified sampling from finite populations
//! - **Genetics**: Allele frequencies in population genetics
//! - **Manufacturing**: Acceptance sampling plans
//!

#[cfg(feature = "simd")]
mod simd;
mod std;

use minarrow::{Bitmask, FloatArray};

use minarrow::enums::error::KernelError;
/// Compute the probability mass function (PMF) for the hypergeometric distribution.
///
/// The hypergeometric distribution models the number of successes in a fixed number of draws
/// from a finite population without replacement. This is the discrete analogue of sampling
/// without replacement, contrasting with the binomial distribution which assumes sampling
/// with replacement or infinite population size.
///
/// ## Mathematical Definition
///
/// The PMF of the hypergeometric distribution is defined as:
///
/// ```text
/// P(X = k) = C(K, k) × C(N-K, n-k) / C(N, n)
/// ```
///
/// where:
/// - `N` (population) is the total population size
/// - `K` (success) is the number of success items in the population
/// - `n` (draws) is the number of items drawn (sample size)
/// - `k` is the observed number of successes in the sample
/// - `C(a, b)` denotes the binomial coefficient "a choose b"
///
/// ## Parameters
///
/// * `k` - Number of observed successes in the sample (non-negative integers)
/// * `population` - Total population size N (positive integer)
/// * `success` - Number of success items in population K (0 ≤ K ≤ N)
/// * `draws` - Number of items drawn n (0 ≤ n ≤ N)
/// * `null_mask` - Optional bitmask for null value handling in Arrow format
/// * `null_count` - Optional count of null values for optimisation
///
/// ## Returns
///
/// Returns a `FloatArray<f64>` containing the PMF values, or a `KernelError` if:
/// - Population size is zero (N = 0)
/// - Success count exceeds population (K > N)
/// - Draw count exceeds population (n > N)
/// - Any parameter is invalid or inconsistent
///
/// ## Applications
///
/// - **Quality control**: Defective items in acceptance sampling
/// - **Ecology**: Capture-recapture population estimation studies
/// - **Card games**: Probability calculations for specific hands
/// - **Survey sampling**: Stratified sampling from finite populations
/// - **Genetics**: Allele frequency analysis in population genetics
/// - **Manufacturing**: Batch testing and acceptance sampling plans
/// - **Auditing**: Sample-based auditing of finite populations
#[inline(always)]
pub fn hypergeometric_pmf(
    k: &[u64],
    population: u64,
    success: u64,
    draws: u64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::hypergeometric_pmf_simd(k, population, success, draws, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::hypergeometric_pmf_std(k, population, success, draws, null_mask, null_count)
    }
}

/// Hypergeometric CDF: F(k) = ∑_{i=0}^k PMF(i)
#[inline(always)]
pub fn hypergeometric_cdf(
    k: &[u64],
    population: u64,
    success: u64,
    draws: u64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::hypergeometric_cdf_std(k, population, success, draws, null_mask, null_count)
}

/// Hypergeometric quantile: Q(p) = smallest k such that CDF(k) ≥ p
#[inline(always)]
pub fn hypergeometric_quantile(
    p: &[f64],
    population: u64,
    success: u64,
    draws: u64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::hypergeometric_quantile_std(p, population, success, draws, null_mask, null_count)
}

/// Zero-allocation variant of [`hypergeometric_pmf`].
///
/// Writes directly to caller-provided output buffer.
/// P(X = k) = C(K,k) × C(N-K, n-k) / C(N,n)
#[inline(always)]
pub fn hypergeometric_pmf_to(
    k: &[u64],
    population: u64,
    success: u64,
    draws: u64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::hypergeometric_pmf_simd_to(
            k, population, success, draws, output, null_mask, null_count,
        )
    }

    #[cfg(not(feature = "simd"))]
    {
        std::hypergeometric_pmf_std_to(k, population, success, draws, output, null_mask, null_count)
    }
}

/// Zero-allocation variant of [`hypergeometric_cdf`].
///
/// Writes directly to caller-provided output buffer.
/// F(k) = ∑_{i=0}^k PMF(i)
#[inline(always)]
pub fn hypergeometric_cdf_to(
    k: &[u64],
    population: u64,
    success: u64,
    draws: u64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::hypergeometric_cdf_std_to(k, population, success, draws, output, null_mask, null_count)
}

/// Zero-allocation variant of [`hypergeometric_quantile`].
///
/// Writes directly to caller-provided output buffer.
/// Q(p) = smallest k such that CDF(k) ≥ p
#[inline(always)]
pub fn hypergeometric_quantile_to(
    p: &[f64],
    population: u64,
    success: u64,
    draws: u64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::hypergeometric_quantile_std_to(
        p, population, success, draws, output, null_mask, null_count,
    )
}

#[cfg(test)]
mod tests {
    use crate::kernels::scientific::distributions::univariate::common::dense_data;

    use super::*;
    use minarrow::{Bitmask, vec64};

    // See `./tests` for the scipy test suite

    // -------- helpers ----------------------------------------------------

    /// Extract data when we *know* the result is dense.

    fn mask_vec(mask: &Bitmask) -> Vec<bool> {
        (0..mask.len()).map(|i| mask.get(i)).collect()
    }

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "assert_close failed: {} vs {} (tol={})",
            a,
            b,
            tol
        )
    }

    // -------- reference data  ------------

    const N_POP: u64 = 20;
    const N_SUC: u64 = 7;
    const N_DRAW: u64 = 12;
    const MIN_K: usize = 7; // min(draws, success)

    /// pmf[k] for k = 0 … 7   (SciPy: stats.hypergeom.pmf)
    const PMF_REF: [f64; 8] = [
        1.031_991_744_066_0474e-4,
        4.334_365_325_077_4e-3,
        4.767_801_857_585_139e-2,
        1.986_584_107_327_141_5e-1,
        3.575_851_393_188_854_7e-1,
        2.860_681_114_551_083_4e-1,
        9.535_603_715_170_278e-2,
        1.021_671_826_625_387e-2,
    ];

    /// cdf[k] = Σ_{i=0}^k pmf[i]
    const CDF_REF: [f64; 8] = [
        1.031_991_744_066_0474e-4,
        4.437_564_499_484_005e-3,
        5.211_558_307_533_54e-2,
        2.507_739_938_080_495_7e-1,
        6.083_591_331_269_35e-1,
        8.944_272_445_820_434e-1,
        9.897_832_817_337_462e-1,
        1.0,
    ];

    // -------- PMF: correctness -------------------------------------------

    #[test]
    fn hypergeom_pmf_reference_values() {
        let k: Vec<u64> = (0..=MIN_K as u64).collect();
        let out = dense_data(hypergeometric_pmf(&k, N_POP, N_SUC, N_DRAW, None, None).unwrap());
        for (calc, expect) in out.iter().zip(PMF_REF.iter()) {
            assert_close(*calc, *expect, 1e-14);
        }
    }

    #[test]
    fn hypergeom_pmf_out_of_range_is_zero() {
        // k = 9 lies outside the support when min_k = 7.
        let out = dense_data(hypergeometric_pmf(&[9], N_POP, N_SUC, N_DRAW, None, None).unwrap());
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn hypergeom_pmf_bulk_vs_scalar() {
        let ks = vec64![0_u64, 3, 5, 7];
        let bulk = dense_data(hypergeometric_pmf(&ks, N_POP, N_SUC, N_DRAW, None, None).unwrap());
        for (i, &k) in ks.iter().enumerate() {
            let scalar =
                dense_data(hypergeometric_pmf(&[k], N_POP, N_SUC, N_DRAW, None, None).unwrap())[0];
            assert_close(bulk[i], scalar, 1e-15);
        }
    }

    // -------- PMF: null-mask propagation ---------------------------------

    #[test]
    fn hypergeom_pmf_mask_propagation() {
        let k = vec64![0_u64, 2, 4, 6];
        let mut mask = Bitmask::new_set_all(4, true);
        unsafe { mask.set_unchecked(1, false) }; // null the 2nd slot

        let arr = hypergeometric_pmf(&k, N_POP, N_SUC, N_DRAW, Some(&mask), Some(1)).unwrap();
        let out_mask = mask_vec(arr.null_mask.as_ref().unwrap());

        assert_eq!(out_mask, vec![true, false, true, true]); // same pattern
        assert!(arr.data[1].is_nan()); // NaN sentinel
    }

    // -------- PMF: invalid parameter handling ----------------------------

    #[test]
    fn hypergeom_pmf_invalid_params() {
        // success > population
        assert!(hypergeometric_pmf(&[0], 10, 11, 5, None, None).is_err());
        // draws > population
        assert!(hypergeometric_pmf(&[0], 10, 5, 12, None, None).is_err());
        // population = 0
        assert!(hypergeometric_pmf(&[0], 0, 0, 0, None, None).is_err());
    }

    // -------- CDF: correctness -------------------------------------------

    #[test]
    fn hypergeom_cdf_reference_values() {
        let k: Vec<u64> = (0..=MIN_K as u64).collect();
        let out = dense_data(hypergeometric_cdf(&k, N_POP, N_SUC, N_DRAW, None, None).unwrap());
        for (calc, expect) in out.iter().zip(CDF_REF.iter()) {
            assert_close(*calc, *expect, 1e-14);
        }
    }

    #[test]
    fn hypergeom_cdf_is_monotone() {
        let k: Vec<u64> = (0..=MIN_K as u64).collect();
        let out = dense_data(hypergeometric_cdf(&k, N_POP, N_SUC, N_DRAW, None, None).unwrap());
        for win in out.windows(2) {
            assert!(win[1] >= win[0]);
        }
        assert_close(*out.last().unwrap(), 1.0, 1e-14);
    }

    // -------- Quantile: correctness --------------------------------------

    #[test]
    fn hypergeom_quantile_reference_values() {
        // p grid spanning [0,1]
        let p = [
            0.0,  // -> -1
            5e-5, // between 0 and first PMF -> 0
            0.05, 0.3, 0.7, 0.95, 1.0, // assorted interior / tail
        ];
        let expect = vec64![-1.0, 0.0, 2.0, 4.0, 5.0, 6.0, 7.0];
        let out =
            dense_data(hypergeometric_quantile(&p, N_POP, N_SUC, N_DRAW, None, None).unwrap());
        for (calc, exp) in out.iter().zip(expect.iter()) {
            assert_eq!(*calc, *exp);
        }
    }

    #[test]
    fn hypergeom_quantile_roundtrip() {
        // CDF -> Quantile should be non-decreasing and “round-trip” for these ks
        let k: Vec<u64> = (0..=MIN_K as u64).collect();
        let cdf = dense_data(hypergeometric_cdf(&k, N_POP, N_SUC, N_DRAW, None, None).unwrap());
        let q =
            dense_data(hypergeometric_quantile(&cdf, N_POP, N_SUC, N_DRAW, None, None).unwrap());
        for (k_i, q_i) in k.iter().zip(q.iter()) {
            assert_eq!(*k_i as f64, *q_i);
        }
    }

    // -------- Quantile: domain / mask / param errors ---------------------

    #[test]
    fn hypergeom_quantile_domain_and_mask() {
        let p = vec64![f64::NAN, -0.1, 0.2, 1.1];
        let mut mask = Bitmask::new_set_all(4, true);
        unsafe { mask.set_unchecked(2, false) }; // hide p = 0.2

        let arr = hypergeometric_quantile(&p, N_POP, N_SUC, N_DRAW, Some(&mask), Some(1)).unwrap();
        let out_mask = mask_vec(arr.null_mask.as_ref().unwrap());

        // Data checks
        assert!(arr.data[0].is_nan());
        assert!(arr.data[1].is_nan());
        assert!(arr.data[2].is_nan()); // masked input
        assert!(arr.data[3].is_nan());

        // Mask propagation
        assert_eq!(out_mask, vec![true, true, false, true]);
    }

    #[test]
    fn hypergeom_quantile_invalid_params() {
        // impossible parameter combos
        assert!(hypergeometric_quantile(&[0.5], 10, 11, 5, None, None).is_err());
        assert!(hypergeometric_quantile(&[0.5], 10, 5, 12, None, None).is_err());
    }
}
