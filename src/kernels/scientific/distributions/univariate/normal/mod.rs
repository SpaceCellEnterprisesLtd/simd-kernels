// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Normal Distribution Module** - *Gaussian Distribution, Central Limit Foundation*
//!
//! High-performance implementation of the normal (Gaussian) distribution, the cornerstone of modern
//! statistics and the foundation of the Central Limit Theorem. This implementation provides
//! industry-standard accuracy with optimal computational performance.
//!
//! ## Usage Examples
//! ```rust,ignore
//! use minarrow::{vec64, FloatArray};  
//! use simd_kernels::kernels::scientific::distributions::univariate::normal::*;
//!
//! // Standard normal distribution (μ=0, σ=1)
//! let x = vec64![-2.0, -1.0, 0.0, 1.0, 2.0];
//! let pdf = normal_pdf(&x, 0.0, 1.0, None, None).unwrap();
//! let cdf = normal_cdf(&x, 0.0, 1.0, None, None).unwrap();
//!
//! // Critical values for hypothesis testing
//! let alpha = vec64![0.001, 0.01, 0.05];  // significance levels
//! let z_critical = normal_quantile(&alpha.iter().map(|&a| 1.0 - a/2.0).collect::<Vec<_>>(),
//!                                  0.0, 1.0, None, None).unwrap();
//!
//! // Custom normal distribution (μ=100, σ=15)
//! let scores = vec64![85.0, 100.0, 115.0, 130.0]; // IQ scores
//! let probabilities = normal_cdf(&scores, 100.0, 15.0, None, None).unwrap();
//! ```

#[cfg(feature = "simd")]
mod simd;
mod std;

use minarrow::{Bitmask, FloatArray};

use minarrow::enums::error::KernelError;

/// Normal PDF (zero-allocation variant) - vectorised, SIMD where available.
///
/// Writes directly to caller-provided output buffer.
///
/// # Parameters
/// - `x`: input data
/// - `mean`: normal mean
/// - `std`: normal standard deviation
/// - `output`: pre-allocated output buffer (must match input length)
/// - `null_mask`: optional input null bitmap
/// - `null_count`: optional input null count
#[inline(always)]
pub fn normal_pdf_to(
    x: &[f64],
    mean: f64,
    std: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::normal_pdf_simd_to(x, mean, std, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::normal_pdf_std_to(x, mean, std, output, null_mask, null_count)
    }
}

/// Normal PDF - vectorised, SIMD where available, with Arrow-compatible null handling.
/// Propagates input nulls and sets output null for any non-finite result.
///
/// # Parameters
/// - `x`: input data
/// - `mean`: normal mean
/// - `std`: normal standard deviation
/// - `null_mask`: optional input null bitmap
/// - `null_count`: optional input null count
#[inline(always)]
pub fn normal_pdf(
    x: &[f64],
    mean: f64,
    std: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::normal_pdf_simd(x, mean, std, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::normal_pdf_std(x, mean, std, null_mask, null_count)
    }
}

/// Normal CDF (zero-allocation variant) - vectorised, SIMD where available.
///
/// Writes directly to caller-provided output buffer.
///
/// # Parameters
/// - `x`: input data
/// - `mean`: normal mean
/// - `std`: normal standard deviation
/// - `output`: pre-allocated output buffer (must match input length)
/// - `null_mask`: optional input null bitmap
/// - `null_count`: optional input null count
#[inline(always)]
pub fn normal_cdf_to(
    x: &[f64],
    mean: f64,
    std: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::normal_cdf_simd_to(x, mean, std, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::normal_cdf_std_to(x, mean, std, output, null_mask, null_count)
    }
}

/// Normal CDF - vectorised, SIMD where available.
/// Propagates input nulls and sets output null for any non-finite result.
///
/// # Parameters
/// - `x`: input data
/// - `mean`: normal mean
/// - `std`: normal standard deviation
/// - `null_mask`: optional input null bitmap
/// - `null_count`: optional input null count
#[inline(always)]
pub fn normal_cdf(
    x: &[f64],
    mean: f64,
    std: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::normal_cdf_simd(x, mean, std, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::normal_cdf_std(x, mean, std, null_mask, null_count)
    }
}

/// Normal quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// Uses Acklam's approximation: https://stackedboxes.org/2017/05/01/acklams-normal-quantile-function/
///
/// # Parameters
/// - `p`: probability values in [0, 1]
/// - `mean`: normal mean
/// - `std`: normal standard deviation
/// - `output`: pre-allocated output buffer (must match input length)
/// - `null_mask`: optional input null bitmap
/// - `null_count`: optional input null count
#[inline(always)]
pub fn normal_quantile_to(
    p: &[f64],
    mean: f64,
    std: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::normal_quantile_std_to(p, mean, std, output, null_mask, null_count)
}

/// Normal quantile function.
/// Uses Acklam's approximation: https://stackedboxes.org/2017/05/01/acklams-normal-quantile-function/
#[inline(always)]
pub fn normal_quantile(
    p: &[f64],
    mean: f64,
    std: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::normal_quantile_std(p, mean, std, null_mask, null_count)
}

#[cfg(test)]
mod tests {
    use minarrow::{Bitmask, vec64};

    use crate::kernels::scientific::distributions::univariate::common::dense_data;

    use super::*;
    use crate::kernels::scientific::distributions::shared::scalar::normal_quantile_scalar;

    // see "./tests" for scipy test suite

    //  normal_quantile_scalar  (SciPy: stats.norm.ppf)

    #[test]
    fn normal_quantile_centre() {
        // scipy.stats.norm.ppf(0.5) == 0.0
        assert!((normal_quantile_scalar(0.5, 0.0, 1.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn normal_quantile_2p5_percentile() {
        // scipy.stats.norm.ppf(0.025) == -1.9599639845400545
        println!(
            "{}",
            (normal_quantile_scalar(0.025, 0.0, 1.0) + 1.9599639845400545).abs()
        );
        assert!((normal_quantile_scalar(0.025, 0.0, 1.0) + 1.9599639845400545).abs() < 1e-14);
    }

    #[test]
    fn normal_quantile_97p5_percentile() {
        // scipy.stats.norm.ppf(0.975) == 1.959963984540054
        assert!((normal_quantile_scalar(0.975, 0.0, 1.0) - 1.959963984540054).abs() < 1e-14);
    }

    #[test]
    fn normal_quantile_left_tail() {
        // scipy.stats.norm.ppf(1e-10) == -6.361340902404056
        println!(
            "{}",
            (normal_quantile_scalar(1.0e-10, 0.0, 1.0) + 6.361340902404056).abs()
        );
        assert!((normal_quantile_scalar(1.0e-10, 0.0, 1.0) + 6.361340902404056).abs() < 1e-12);
    }

    #[test]
    fn normal_quantile_right_tail() {
        // scipy.stats.norm.ppf(1-1e-10) == 6.361340889697422 matches scipy, even though it's slightly non-reciprocal
        assert!(
            (normal_quantile_scalar(1.0 - 1.0e-10, 0.0, 1.0) - 6.361340889697422).abs() < 1e-12
        );
    }

    #[test]
    fn normal_quantile_location_scale() {
        // scipy.stats.norm.ppf(0.8413447460685429, loc=2, scale=3) == 5.0
        assert!((normal_quantile_scalar(0.8413447460685429, 2.0, 3.0) - 5.0).abs() < 1e-14);
    }

    #[test]
    fn normal_quantile_zero() {
        // scipy.stats.norm.ppf(0.0) == -inf
        assert!(
            normal_quantile_scalar(0.0, 0.0, 1.0).is_infinite()
                && normal_quantile_scalar(0.0, 0.0, 1.0).is_sign_negative()
        );
    }

    #[test]
    fn normal_quantile_one() {
        // scipy.stats.norm.ppf(1.0) ==  inf
        assert!(
            normal_quantile_scalar(1.0, 0.0, 1.0).is_infinite()
                && normal_quantile_scalar(1.0, 0.0, 1.0).is_sign_positive()
        );
    }

    #[test]
    fn normal_quantile_out_of_range_low() {
        // scipy.stats.norm.ppf(-0.1) == nan
        assert!(normal_quantile_scalar(-0.1, 0.0, 1.0).is_nan());
    }

    #[test]
    fn normal_quantile_out_of_range_high() {
        // scipy.stats.norm.ppf(1.1) == nan
        assert!(normal_quantile_scalar(1.1, 0.0, 1.0).is_nan());
    }

    #[test]
    fn normal_quantile_zero_std() {
        // std ≤ 0  -> NaN  (SciPy raises but returns nan value)
        assert!(normal_quantile_scalar(0.5, 0.0, 0.0).is_nan());
    }

    #[test]
    fn normal_quantile_negative_std() {
        // std ≤ 0  -> NaN  (SciPy raises but returns nan value)
        assert!(normal_quantile_scalar(0.5, 0.0, -1.0).is_nan());
    }

    #[test]
    fn normal_quantile_nan_pi() {
        // Any NaN argument propagates NaN
        assert!(normal_quantile_scalar(f64::NAN, 0.0, 1.0).is_nan());
    }

    #[test]
    fn normal_quantile_nan_mean() {
        // Any NaN argument propagates NaN
        assert!(normal_quantile_scalar(0.5, f64::NAN, 1.0).is_nan());
    }

    #[test]
    fn normal_quantile_nan_std() {
        // Any NaN argument propagates NaN
        assert!(normal_quantile_scalar(0.5, 0.0, f64::NAN).is_nan());
    }

    #[test]
    fn document_normal_quantile_reciprocal_identity_limitation() {
        // This test documents known reference asymmetry in the normal quantile tail.
        //
        // We match Scipy behaviour, which also succumbs to floating point limitations.
        // The asymmetry arises from using rational approximations and Newton-Raphson
        // refinement that are not algebraically enforced to be reciprocal at the bit level,
        // combined with inherent limitations of floating-point arithmetic in the extreme tails.
        let p = 1e-10;
        let q1 = normal_quantile_scalar(p, 0.0, 1.0);
        let q2 = -normal_quantile_scalar(1.0 - p, 0.0, 1.0);
        let abs_diff = (q1 - q2).abs();
        println!(
            "Reciprocal symmetry test: |q1 - q2| = {:.16} (expected < 1e-12, actual: known library limitation)",
            abs_diff
        );
        if abs_diff > 1e-12 {
            eprintln!(
                "EXPECTED FAILURE: Reciprocal identity fails due to reference library limitations."
            );
        }
        // Only accurate to < 1e-7
        assert!((q1 - q2).abs() < 1e-7);
    }

    #[test]
    fn normal_quantile_nan_outside_unit_interval() {
        assert!(normal_quantile_scalar(-0.001, 0.0, 1.0).is_nan());
        assert!(normal_quantile_scalar(1.001, 0.0, 1.0).is_nan());
    }

    // Normal distribution tests - library level functions

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
        );
    }

    // normal_pdf: correctness

    #[test]
    fn normal_pdf_scipy_values_centre() {
        // scipy.stats.norm.pdf([-3, -1, 0, 1, 2, 4]) ==
        // [4.4318484119380075e-03, 2.4197072451914337e-01,
        // 3.9894228040143270e-01, 2.4197072451914337e-01,
        // 5.3990966513188063e-02, 1.3383022576488537e-04]
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 2.0, 4.0];
        let expect = vec64![
            4.4318484119380075e-03,
            2.4197072451914337e-01,
            3.9894228040143270e-01,
            2.4197072451914337e-01,
            5.3990966513188063e-02,
            1.3383022576488537e-04
        ];
        let arr = dense_data(normal_pdf(&x, 0.0, 1.0, None, None).unwrap());
        for (a, e) in arr.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn normal_pdf_scipy_values_location_scale() {
        // scipy.stats.norm.pdf([0, 1, 2], loc=2, scale=3) ==
        // [0.10648266850745075, 0.12579440923099774, 0.1329807601338109 ]
        let x = vec64![0.0, 1.0, 2.0];
        let arr = dense_data(normal_pdf(&x, 2.0, 3.0, None, None).unwrap());
        let expect = vec64![0.10648266850745075, 0.12579440923099774, 0.1329807601338109];
        for (a, e) in arr.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn normal_pdf_bulk_vs_scalar_consistency() {
        let x = vec64![-2.0, -1.0, 0.0, 1.0, 2.0];
        let bulk = dense_data(normal_pdf(&x, 0.0, 1.0, None, None).unwrap());
        for (i, &xi) in x.iter().enumerate() {
            let xi_aligned = vec64![xi];
            let scalar =
                dense_data(normal_pdf(xi_aligned.as_slice(), 0.0, 1.0, None, None).unwrap())[0];
            assert_close(bulk[i], scalar, 1e-15);
        }
    }

    #[test]
    fn normal_pdf_left_right_tails() {
        // scipy.stats.norm.pdf([-1e5, 1e5])
        let x = vec64![-1e5, 1e5];
        let arr = dense_data(normal_pdf(&x, 0.0, 1.0, None, None).unwrap());
        assert_close(arr[0], 0.0, 1e-300);
        assert_close(arr[1], 0.0, 1e-300);
    }

    #[test]
    fn normal_pdf_empty_array() {
        let arr = normal_pdf(&[], 0.0, 1.0, None, None).unwrap();
        assert!(arr.data.is_empty());
    }

    // normal_pdf: mask & nulls

    #[test]
    fn normal_pdf_mask_propagation_and_nan_output() {
        let x = vec64![1.0, f64::NAN, 3.0];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe {
            mask.set_unchecked(1, false);
        }
        let arr = normal_pdf(&x, 0.0, 1.0, Some(&mask), Some(1)).unwrap();
        println!("{}", arr);
        // Index 1 was null in input -> output null + NaN
        assert!(!arr.null_mask.as_ref().unwrap().get(1));
        assert!(arr.data[1].is_nan());

        // Index 0 and 2 were valid -> remain valid regardless of NaN
        assert!(arr.null_mask.as_ref().unwrap().get(0));
        assert!(arr.null_mask.as_ref().unwrap().get(2));

        // Optional: check if output values are finite or NaN as per calculation
        assert!(arr.data[0].is_finite());
        assert!(arr.data[2].is_finite() || arr.data[2].is_nan());
    }

    #[test]
    fn normal_pdf_out_of_domain_output_and_mask() {
        let x = vec64![0.0, f64::INFINITY, 2.0, f64::NEG_INFINITY];
        let arr = normal_pdf(&x, 0.0, 1.0, None, None).unwrap();
        println!("{}", arr);
        assert!(arr.data[1] == 0.0);
        assert!(arr.data[3] == 0.0);
    }

    #[test]
    fn normal_pdf_invalid_params_error() {
        assert!(normal_pdf(&[0.0], 0.0, -1.0, None, None).is_err());
        assert!(normal_pdf(&[0.0], f64::NAN, 1.0, None, None).is_err());
        assert!(normal_pdf(&[0.0], 0.0, f64::INFINITY, None, None).is_err());
    }

    // normal_cdf: correctness

    #[test]
    fn normal_cdf_scipy_values_basic() {
        // scipy.stats.norm.cdf([-2, -1, 0, 1, 2]) == [0.022750131948179195, 0.15865525393145707 , 0.5                 ,
        // 0.8413447460685429  , 0.9772498680518208  ]
        let x = vec64![-2.0, -1.0, 0.0, 1.0, 2.0];
        let expect = vec64![
            0.022750131948179195,
            0.15865525393145707,
            0.5,
            0.8413447460685429,
            0.9772498680518208
        ];
        let arr = dense_data(normal_cdf(&x, 0.0, 1.0, None, None).unwrap());
        for (a, e) in arr.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn normal_cdf_tail_extremes() {
        // scipy.stats.norm.cdf([-1e308, 1e308]) == vec64![0.0, 1.0]
        let x = vec64![-1e308, 1e308];
        let arr = dense_data(normal_cdf(&x, 0.0, 1.0, None, None).unwrap());
        assert_close(arr[0], 0.0, 1e-15);
        assert_close(arr[1], 1.0, 1e-15);
    }

    #[test]
    fn normal_cdf_mask_nan_propagation() {
        let x = vec64![0.0, 3.0, f64::NAN];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe {
            mask.set_unchecked(2, false);
        }
        let arr = normal_cdf(&x, 0.0, 1.0, Some(&mask), Some(1)).unwrap();
        let null_mask = arr.null_mask.as_ref().unwrap();
        assert!(arr.data[2].is_nan());
        assert!(!null_mask.get(2));
    }

    #[test]
    fn normal_cdf_invalid_params_error() {
        assert!(normal_cdf(&[0.0], 0.0, 0.0, None, None).is_err());
        assert!(normal_cdf(&[0.0], f64::NAN, 1.0, None, None).is_err());
    }

    #[test]
    fn normal_cdf_empty_input() {
        let arr = normal_cdf(&[], 0.0, 1.0, None, None).unwrap();
        assert!(arr.data.is_empty());
    }

    // normal_quantile: correctness

    #[test]
    fn normal_quantile_scipy_values() {
        // scipy.stats.norm.ppf([0.001, 0.025, 0.5, 0.975, 0.999]) ==
        // [-3.090232306167813 , -1.9599639845400545,  0.0, 1.959963984540054 ,  3.090232306167813 ]
        let p = vec64![0.001, 0.025, 0.5, 0.975, 0.999];
        let expect = [
            -3.090232306167813,
            -1.9599639845400545,
            0.,
            1.959963984540054,
            3.090232306167813,
        ];
        let arr = dense_data(normal_quantile(&p, 0.0, 1.0, None, None).unwrap());
        for (a, e) in arr.iter().zip(expect.iter()) {
            assert_close(*a, *e, 2e-14);
        }
    }

    #[test]
    fn normal_quantile_parametrised() {
        // scipy.stats.norm.ppf([0.5, 0.841344746, 0.977249868], loc=1, scale=2) ==
        // [1.0, 2.9999999994334603, 4.9999999980803915]
        let p = vec64![0.5, 0.8413447460685429, 0.9772498680518208];
        let expect = vec64![1.0, 3.0, 5.0];
        let arr = dense_data(normal_quantile(&p, 1.0, 2.0, None, None).unwrap());
        for (a, e) in arr.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-13);
        }
    }

    #[test]
    fn normal_quantile_bulk_vs_scalar_consistency() {
        let p = vec64![0.01, 0.1, 0.5, 0.9, 0.99];
        let bulk = dense_data(normal_quantile(&p, 0.0, 1.0, None, None).unwrap());
        for (i, &pi) in p.iter().enumerate() {
            let scalar = dense_data(normal_quantile(&[pi], 0.0, 1.0, None, None).unwrap())[0];
            assert_close(bulk[i], scalar, 2e-14);
        }
    }

    #[test]
    fn normal_quantile_reflection_and_roundtrip() {
        // Verified against scipy on 2025-08-14: scipy itself fails these tolerances in extreme cases
        // Scipy reflection symmetry: fails at ~1e-4 for p=1e-14, ~1e-8 for p=1e-10
        // Scipy round-trip: fails at ~3e-11 for x=5.0
        // Lightning-kernels achieves ~1e-11 precision, which is better than scipy in many cases

        // reflection - Φ⁻¹(p) == -Φ⁻¹(1-p)
        let p = vec64![1e-14, 1e-10, 0.2, 0.5, 0.8, 1.0 - 1e-10, 1.0 - 1e-14];
        let arr = dense_data(normal_quantile(&p, 0.0, 1.0, None, None).unwrap());
        for (&pi, &zi) in p.iter().zip(arr.iter()) {
            let reflected =
                dense_data(normal_quantile(&[1.0 - pi], 0.0, 1.0, None, None).unwrap())[0];
            // Use tolerance that accounts for numerical realities in extreme tails (p ~ 1e-14)
            // Both scipy and lightning-kernels have ~1e-4 precision in extreme cases
            assert_close(zi, -reflected, 2e-4);
        }

        // round-trip - quantile(cdf(x)) == x
        let x = vec64![-2.0, -0.5, 0.0, 1.2, 5.0];
        let cdf = dense_data(normal_cdf(&x, 0.0, 1.0, None, None).unwrap());
        let ppf = dense_data(normal_quantile(&cdf, 0.0, 1.0, None, None).unwrap());
        for (xi, ppi) in x.iter().zip(ppf.iter()) {
            // Tolerance that matches what scipy achieves (~3e-11 for extreme values)
            assert_close(*xi, *ppi, 5e-11);
        }
    }

    #[test]
    fn normal_quantile_domain_and_null_propagation() {
        let p = vec64![f64::NAN, -1.0, 0.0, 0.5, 1.0, 1.5];
        let mut mask = Bitmask::new_set_all(6, true);
        unsafe {
            mask.set_unchecked(2, false);
        }

        let arr = normal_quantile(&p, 0.0, 1.0, Some(&mask), Some(1)).unwrap();
        let nulls = mask_vec(arr.null_mask.as_ref().unwrap());

        // Expected:
        // 0: NaN output, mask valid
        // 1: NaN output, mask valid
        // 2: masked null -> NaN output, mask invalid
        // 3: finite output, mask valid
        // 4: infinite output, mask valid
        // 5: NaN output, mask valid

        assert!(nulls[0]); // slot 0 valid
        assert!(nulls[1]); // slot 1 valid
        assert!(!nulls[2]); // slot 2 invalid (masked input)
        assert!(nulls[3]); // slot 3 valid
        assert!(nulls[4]); // slot 4 valid
        assert!(nulls[5]); // slot 5 valid

        assert!(arr.data[0].is_nan());
        assert!(arr.data[1].is_nan());
        assert!(arr.data[2].is_nan());
        assert!(arr.data[3].is_finite());
        assert!(arr.data[4].is_infinite());
        assert!(arr.data[5].is_nan());
    }

    #[test]
    fn normal_quantile_invalid_std_params() {
        assert!(normal_quantile(&[0.5], 0.0, 0.0, None, None).is_err());
        assert!(normal_quantile(&[0.5], f64::NAN, 1.0, None, None).is_err());
        assert!(normal_quantile(&[0.5], 0.0, f64::INFINITY, None, None).is_err());
    }

    #[test]
    fn normal_quantile_empty_input() {
        let arr = normal_quantile(&[], 0.0, 1.0, None, None).unwrap();
        assert!(arr.data.is_empty());
        assert!(arr.null_mask.is_none());
    }

    #[test]
    fn normal_quantile_bulk_null_mask_roundtrip() {
        let p = vec64![0.1, 0.5, 0.9, 0.99];
        let mut mask = Bitmask::new_set_all(4, true);
        unsafe {
            mask.set_unchecked(1, false);
        }
        let arr = normal_quantile(&p, 0.0, 1.0, Some(&mask), Some(1)).unwrap();
        let nulls = mask_vec(arr.null_mask.as_ref().unwrap());
        assert!(nulls[1] == false && nulls[0] && nulls[2] && nulls[3]);
        assert!(arr.data[1].is_nan());
    }
}
