// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

#[cfg(feature = "simd")]
mod simd;
#[cfg(not(feature = "simd"))]
mod std;

use minarrow::{Bitmask, FloatArray};

use minarrow::enums::error::KernelError;

/// Discrete uniform PMF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn discrete_uniform_pmf_to(
    k: &[i64],
    low: i64,
    high: i64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::discrete_uniform_pmf_simd_to(k, low, high, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::discrete_uniform_pmf_std_to(k, low, high, output, null_mask, null_count)
    }
}

/// Discrete uniform CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn discrete_uniform_cdf_to(
    k: &[i64],
    low: i64,
    high: i64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::discrete_uniform_cdf_simd_to(k, low, high, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::discrete_uniform_cdf_std_to(k, low, high, output, null_mask, null_count)
    }
}

/// Discrete uniform quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn discrete_uniform_quantile_to(
    p: &[f64],
    low: i64,
    high: i64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::discrete_uniform_quantile_simd_to(p, low, high, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::discrete_uniform_quantile_std_to(p, low, high, output, null_mask, null_count)
    }
}

/// Discrete uniform PMF.
/// P(X=k) = 1/(high−low+1) for k ∈ [low, high], else 0
#[inline(always)]
pub fn discrete_uniform_pmf(
    k: &[i64],
    low: i64,
    high: i64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::discrete_uniform_pmf_simd(k, low, high, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::discrete_uniform_pmf_std(k, low, high, null_mask, null_count)
    }
}

/// Discrete uniform CDF (lower tail, inclusive).
/// F(k) = 0 for k < low; F(k) = 1 for k ≥ high; else F(k) = (k − low + 1) / (high − low + 1).
#[inline(always)]
pub fn discrete_uniform_cdf(
    k: &[i64],
    low: i64,
    high: i64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::discrete_uniform_cdf_simd(k, low, high, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::discrete_uniform_cdf_std(k, low, high, null_mask, null_count)
    }
}

/// Discrete uniform quantile (lower tail).
/// Returns the smallest integer k ∈ [low, high] such that F(k) ≥ p.
/// For p ≤ 0 -> low; for p ≥ 1 -> high.
/// Output is returned as f64 (integer-valued) to match FloatArray.
#[inline(always)]
pub fn discrete_uniform_quantile(
    p: &[f64],
    low: i64,
    high: i64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::discrete_uniform_quantile_simd(p, low, high, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::discrete_uniform_quantile_std(p, low, high, null_mask, null_count)
    }
}

#[cfg(test)]
mod discrete_uniform_tests {
    use crate::kernels::scientific::distributions::univariate::common::dense_data;

    use super::*;
    use minarrow::{Bitmask, vec64};

    fn mask_vec(mask: &Bitmask) -> Vec<bool> {
        (0..mask.len()).map(|i| mask.get(i)).collect()
    }
    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol,
            "assert_close failed: {a} vs {b}  (tol={tol})"
        );
    }

    // See `./tests` for the scipy test suite

    // ---------- PMF ----------

    #[test]
    fn du_pmf_basic_values() {
        // Support = {2,3,4,5,6}, use high=7 (upper-exclusive)
        let (low, high) = (2_i64, 7_i64);
        let p = 1.0 / 5.0;
        let k = vec64![1i64, 2, 3, 4, 5, 6, 7];
        let out = dense_data(discrete_uniform_pmf(&k, low, high, None, None).unwrap());
        let expect = [0.0, p, p, p, p, p, 0.0];
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn du_pmf_basic_values_with_negatives() {
        // Support = {-3,-2,-1,0,1}, high=2
        let (low, high) = (-3_i64, 2_i64);
        let p = 1.0 / 5.0;
        let k = vec64![-4i64, -3, -2, -1, 0, 1, 2];
        let out = dense_data(discrete_uniform_pmf(&k, low, high, None, None).unwrap());
        let expect = [0.0, p, p, p, p, p, 0.0];
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn du_pmf_sum_to_one() {
        // Want 21 outcomes: 10..=30  -> use high=31
        let (low, high) = (10_i64, 31_i64);
        let k: Vec<i64> = (10..=30).collect();
        let pmf = dense_data(discrete_uniform_pmf(&k, low, high, None, None).unwrap());
        let sum: f64 = pmf.iter().sum();
        assert_close(sum, 1.0, 1e-15);
    }

    #[test]
    fn du_pmf_bulk_vs_scalar_consistency() {
        // Support = {100..=105} -> high=106
        let (low, high) = (100_i64, 106_i64);
        let ks = vec64![99i64, 101, 107];
        let bulk = dense_data(discrete_uniform_pmf(&ks, low, high, None, None).unwrap());
        for (i, &ki) in ks.iter().enumerate() {
            let scalar = dense_data(discrete_uniform_pmf(&[ki], low, high, None, None).unwrap())[0];
            assert_close(bulk[i], scalar, 1e-15);
        }
    }

    #[test]
    fn du_pmf_null_mask_propagation() {
        // Support = {5,6} -> high=7
        let k = vec64![5i64, 6, 7];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe {
            mask.set_unchecked(1, false);
        }
        let arr = discrete_uniform_pmf(&k, 5, 7, Some(&mask), Some(1)).unwrap();
        let nulls = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(nulls, vec![true, false, true]);
        assert!(arr.data[1].is_nan());
    }

    #[test]
    fn du_pmf_invalid_low_ge_high() {
        assert!(discrete_uniform_pmf(&[0], 5, 5, None, None).is_err());
        assert!(discrete_uniform_pmf(&[0], 5, 4, None, None).is_err());
    }

    #[test]
    fn du_pmf_single_point_range() {
        // Single outcome {42} -> low=42, high=43 (upper-exclusive)
        let (low, high) = (42_i64, 43_i64);
        let ks = vec64![41i64, 42, 43];
        let out = dense_data(discrete_uniform_pmf(&ks, low, high, None, None).unwrap());
        assert_eq!(out, vec64![0.0, 1.0, 0.0]);
    }

    #[test]
    fn du_pmf_empty_input() {
        let arr = discrete_uniform_pmf(&[], 0, 10, None, None).unwrap();
        assert!(arr.data.is_empty());
        assert!(arr.null_mask.is_none());
    }

    // ---------- CDF ----------

    #[test]
    fn du_cdf_basic_values() {
        // Support = {2,3,4,5,6} -> high=7
        let (low, high) = (2_i64, 7_i64);
        let k = vec64![1i64, 2, 3, 4, 5, 6, 7];
        let out = dense_data(discrete_uniform_cdf(&k, low, high, None, None).unwrap());
        let expect = [0.0, 1.0 / 5.0, 2.0 / 5.0, 3.0 / 5.0, 4.0 / 5.0, 1.0, 1.0];
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn du_cdf_with_negatives() {
        // Support = {-3,-2,-1,0,1} -> high=2
        let (low, high) = (-3_i64, 2_i64);
        let k = vec64![-4i64, -3, -2, -1, 0, 1, 2];
        let out = dense_data(discrete_uniform_cdf(&k, low, high, None, None).unwrap());
        let expect = [0.0, 1.0 / 5.0, 2.0 / 5.0, 3.0 / 5.0, 4.0 / 5.0, 1.0, 1.0];
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn du_cdf_null_mask_propagation() {
        // Support = {5,6} -> high=7
        let k = vec64![4i64, 5, 6];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe {
            mask.set_unchecked(0, false);
        }
        let arr = discrete_uniform_cdf(&k, 5, 7, Some(&mask), Some(1)).unwrap();
        let nulls = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(nulls, vec![false, true, true]);
        assert!(arr.data[0].is_nan());
    }

    #[test]
    fn du_cdf_empty_input() {
        let arr = discrete_uniform_cdf(&[], 0, 10, None, None).unwrap();
        assert!(arr.data.is_empty());
        assert!(arr.null_mask.is_none());
    }

    // ---------- Quantile ----------

    #[test]
    fn du_quantile_basic_values() {
        // Support = {2,3,4,5,6} -> high=7, N=5
        let (low, high) = (2_i64, 7_i64);
        let p = vec![0.0, 0.2, 0.2000001, 0.4, 0.6, 0.8, 1.0];
        // SciPy randint: Q(0) = low-1; (0,0.2]->2; (0.2,0.4]->3; (0.4,0.6]->4; (0.6,0.8]->5; (0.8,1]->6
        let out = dense_data(discrete_uniform_quantile(&p, low, high, None, None).unwrap());
        let expect = vec64![1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0];
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 0.0);
        }
    }

    #[test]
    fn du_quantile_with_negatives() {
        // Support = {-3,-2,-1,0,1} -> high=2, N=5
        let (low, high) = (-3_i64, 2_i64);
        let p = vec![0.0, 0.01, 0.2, 0.40001, 0.8, 1.0];
        let out = dense_data(discrete_uniform_quantile(&p, low, high, None, None).unwrap());
        // thresholds at 0.2,0.4,0.6,0.8 ; Q(0)=low-1=-4
        let expect = vec64![-4.0, -3.0, -3.0, -1.0, 0.0, 1.0];
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 0.0);
        }
    }

    #[test]
    fn du_quantile_clips_bounds() {
        // Support = {10,11,12} -> high=13, N=3
        let (low, high) = (10_i64, 13_i64);
        let p = vec![-0.5, 0.0, 0.0001, 0.3334, 0.6667, 1.0, 1.5];
        let out = dense_data(discrete_uniform_quantile(&p, low, high, None, None).unwrap());
        // steps at 1/3, 2/3, 1; Q(0)=low-1=9; p<0 clipped to 0
        let expect = vec64![9.0, 9.0, 10.0, 11.0, 12.0, 12.0, 12.0];
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 0.0);
        }
    }

    #[test]
    fn du_quantile_null_mask_propagation() {
        // Support = {0,1} -> high=2
        let p = vec![0.1, 0.5, 0.9];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe {
            mask.set_unchecked(2, false);
        }
        let arr = discrete_uniform_quantile(&p, 0, 2, Some(&mask), Some(1)).unwrap();
        let nulls = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(nulls, vec![true, true, false]);
        assert!(arr.data[2].is_nan());
    }
}
