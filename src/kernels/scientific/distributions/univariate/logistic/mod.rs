//! # Logistic Distribution
//!
//! The logistic distribution is a continuous probability distribution with applications in
//! logistic regression, survival analysis, and neural networks. It resembles the normal
//! distribution but has heavier tails and a simple closed-form CDF.
//!
//! ## Mathematical Definition
//!
//! Parameterised by location μ and scale s > 0:
//! - **PDF**: f(x; μ, s) = exp(-(x-μ)/s) / [s(1 + exp(-(x-μ)/s))²]
//! - **CDF**: F(x; μ, s) = 1 / (1 + exp(-(x-μ)/s))
//! - **Quantile**: Q(p; μ, s) = μ + s ln(p/(1-p))
//!
//! ## Applications
//!
//! - **Machine learning**: Logistic regression and neural network activations
//! - **Survival analysis**: Proportional hazards modelling
//! - **Growth modelling**: S-curve population and economic growth
//! - **Marketing**: Customer conversion and adoption curves

// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

#[cfg(feature = "simd")]
mod simd;
#[cfg(not(feature = "simd"))]
mod std;

use minarrow::{Bitmask, FloatArray};

use minarrow::enums::error::KernelError;

/// Compute the probability density function (PDF) for the logistic distribution.
///
/// The logistic distribution is a continuous probability distribution that closely resembles
/// the normal distribution but has heavier tails and a simple closed-form cumulative distribution
/// function. It is widely used in logistic regression, neural networks, and survival analysis.
///
/// ## Mathematical Definition
///
/// The PDF of the logistic distribution is defined as:
///
/// ```text
/// f(x; μ, s) = exp(-(x-μ)/s) / [s × (1 + exp(-(x-μ)/s))²]
/// ```
///
/// where:
/// - `μ` (location) is the location parameter (mean, median, and mode)
/// - `s` (scale) is the scale parameter, with `s > 0`
///
/// ## Parameters
///
/// * `x` - Input values to evaluate the PDF at
/// * `location` - Location parameter μ (mean, median, and mode of the distribution)
/// * `scale` - Scale parameter s, must be positive (s > 0)
/// * `null_mask` - Optional bitmask for null value handling in Arrow format
/// * `null_count` - Optional count of null values for optimisation
///
/// ## Returns
///
/// Returns a `FloatArray<f64>` containing the PDF values, or a `KernelError` if:
/// - Scale parameter is non-positive (s ≤ 0)
/// - Location parameter is not finite (NaN or infinite)
/// - Scale parameter is not finite (NaN or infinite)
///
/// ## Examples
///
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::logistic::logistic_pdf;
/// use minarrow::vec64;
///
/// // Standard logistic distribution (μ=0, s=1)
/// let x = vec64![-2.0, -1.0, 0.0, 1.0, 2.0];
/// let pdf = logistic_pdf(&x, 0.0, 1.0, None, None).unwrap();
/// ```
///
/// ## Applications
///
/// - **Machine learning**: Logistic regression, neural network activations
/// - **Survival analysis**: Proportional hazards and time-to-event modelling
/// - **Growth modelling**: Population growth, technology adoption curves
/// - **Economics**: Choice modelling and discrete choice analysis
/// - **Marketing**: Customer conversion probability modelling
#[inline(always)]
pub fn logistic_pdf(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::logistic_pdf_simd(x, location, scale, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::logistic_pdf_std(x, location, scale, null_mask, null_count)
    }
}

/// Logistic CDF: F(x|μ, s) = 1 / (1 + exp(-(x-μ)/s))
#[inline(always)]
pub fn logistic_cdf(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::logistic_cdf_simd(x, location, scale, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::logistic_cdf_std(x, location, scale, null_mask, null_count)
    }
}

/// Logistic quantile (inverse CDF) function.
///
/// For a given probability `p` in (0,1), the quantile is:
///     Q(p) = location + scale * ln(p / (1 - p))
///
/// # Arguments
/// - `p`: Probabilities (must all be in (0,1)).
/// - `location`: Location parameter.
/// - `scale`: Scale parameter (must be positive).
///
/// # Returns
/// - `FloatArray<f64>` of quantiles.
/// - Returns error if any argument is invalid or out of range.
#[inline(always)]
pub fn logistic_quantile(
    p: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::logistic_quantile_simd(p, location, scale, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::logistic_quantile_std(p, location, scale, null_mask, null_count)
    }
}

// Zero-allocation variants

/// Logistic PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// f(x|μ, s) = exp(-(x-μ)/s) / [s × (1 + exp(-(x-μ)/s))²]
#[inline(always)]
pub fn logistic_pdf_to(
    x: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::logistic_pdf_simd_to(x, location, scale, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::logistic_pdf_std_to(x, location, scale, output, null_mask, null_count)
    }
}

/// Logistic CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// F(x|μ, s) = 1 / (1 + exp(-(x-μ)/s))
#[inline(always)]
pub fn logistic_cdf_to(
    x: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::logistic_cdf_simd_to(x, location, scale, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::logistic_cdf_std_to(x, location, scale, output, null_mask, null_count)
    }
}

/// Logistic quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// Q(p) = location + scale * ln(p / (1 - p))
#[inline(always)]
pub fn logistic_quantile_to(
    p: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::logistic_quantile_simd_to(p, location, scale, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::logistic_quantile_std_to(p, location, scale, output, null_mask, null_count)
    }
}

#[cfg(test)]
mod tests {

    use minarrow::{Bitmask, vec64};

    use crate::kernels::scientific::distributions::univariate::common::dense_data;

    use super::*;

    // see "./tests" for scipy test suite

    // helpers

    fn mask_vec(mask: &Bitmask) -> Vec<bool> {
        (0..mask.len()).map(|i| mask.get(i)).collect()
    }

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "assert_close failed: {a} vs {b} (tol={tol})"
        );
    }

    // PDF

    #[test]
    fn logistic_pdf_reference_values() {
        // scipy.stats.logistic.pdf([-4, -2, 0, 2, 4]) =>
        // [0.017662706213291118, 0.10499358540350652, 0.25,
        //  0.10499358540350652, 0.017662706213291118]
        let x = vec64![-4.0, -2.0, 0.0, 2.0, 4.0];
        let expect = vec64![
            0.017662706213291118,
            0.10499358540350652,
            0.25,
            0.10499358540350652,
            0.017662706213291118
        ];
        let arr = dense_data(logistic_pdf(&x, 0.0, 1.0, None, None).unwrap());
        for (a, e) in arr.iter().zip(expect.iter()) {
            assert_close(*a, *e, 2e-15);
        }
    }

    #[test]
    fn logistic_pdf_location_scale() {
        // scipy.stats.logistic.pdf([0,1,2], loc=2, scale=3)
        // Verified against scipy on 2025-08-14: [0.07471912996707621, 0.08106072033492358, 0.08333333333333333]
        let x = vec64![0.0, 1.0, 2.0];
        let expect = vec64![
            0.07471912996707621,
            0.08106072033492358,
            0.08333333333333333
        ];
        let arr = dense_data(logistic_pdf(&x, 2.0, 3.0, None, None).unwrap());
        for (a, e) in arr.iter().zip(expect.iter()) {
            assert_close(*a, *e, 2e-15);
        }
    }

    #[test]
    fn logistic_pdf_bulk_vs_scalar() {
        let x = vec64![-3.0, -1.5, 0.0, 1.5, 3.0];
        let bulk = dense_data(logistic_pdf(&x, 0.0, 1.0, None, None).unwrap());
        for (i, &xi) in x.iter().enumerate() {
            let xi_aligned = vec64![xi];
            let scalar =
                dense_data(logistic_pdf(xi_aligned.as_slice(), 0.0, 1.0, None, None).unwrap())[0];
            assert_close(bulk[i], scalar, 2e-15);
        }
    }

    #[test]
    fn logistic_pdf_mask_propagation_nan() {
        let x = vec64![1.0, f64::NAN, 3.0];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe { mask.set_unchecked(1, false) }; // make element 1 null
        let arr = logistic_pdf(&x, 0.0, 1.0, Some(&mask), Some(1)).unwrap();
        let nulls = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(nulls, vec![true, false, true]);
        assert!(arr.data[1].is_nan());
    }

    #[test]
    fn logistic_pdf_invalid_params() {
        assert!(logistic_pdf(&[0.0], 0.0, 0.0, None, None).is_err());
        assert!(logistic_pdf(&[0.0], f64::NAN, 1.0, None, None).is_err());
        assert!(logistic_pdf(&[0.0], 0.0, f64::INFINITY, None, None).is_err());
    }

    // CDF

    #[test]
    fn logistic_cdf_reference_values() {
        // scipy.stats.logistic.cdf([-4,-2,0,2,4]) ==
        // [0.01798620996209156, 0.11920292202211755,
        //  0.5, 0.8807970779778823, 0.9820137900379085]
        let x = vec64![-4.0, -2.0, 0.0, 2.0, 4.0];
        let expect = [
            0.01798620996209156,
            0.11920292202211755,
            0.5,
            0.8807970779778823,
            0.9820137900379085,
        ];
        let arr = dense_data(logistic_cdf(&x, 0.0, 1.0, None, None).unwrap());
        for (a, e) in arr.iter().zip(expect.iter()) {
            assert_close(*a, *e, 2e-15);
        }
    }

    #[test]
    fn logistic_cdf_tail_extremes() {
        let x = vec64![-1e8, 1e8];
        let arr = dense_data(logistic_cdf(&x, 0.0, 1.0, None, None).unwrap());
        assert_close(arr[0], 0.0, 1e-15); // underflow to 0
        assert_close(arr[1], 1.0, 1e-15); // overflow to 1
    }

    #[test]
    fn logistic_cdf_mask_nan() {
        let x = vec64![0.0, 3.0, f64::NAN];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe { mask.set_unchecked(2, false) };
        let arr = logistic_cdf(&x, 0.0, 1.0, Some(&mask), Some(1)).unwrap();
        assert!(arr.data[2].is_nan());
        assert!(!arr.null_mask.as_ref().unwrap().get(2));
    }

    #[test]
    fn logistic_cdf_invalid_params() {
        assert!(logistic_cdf(&[0.0], 0.0, 0.0, None, None).is_err());
        assert!(logistic_cdf(&[0.0], f64::NAN, 1.0, None, None).is_err());
    }

    #[test]
    fn logistic_cdf_empty() {
        let arr = logistic_cdf(&[], 0.0, 1.0, None, None).unwrap();
        assert!(arr.data.is_empty());
    }

    // ──────────────────────────── quantile ───────────────────────────────

    #[test]
    fn logistic_quantile_reference_values() {
        // scipy.stats.logistic.ppf([0.001,0.025,0.5,0.975,0.999]) ==
        // [-6.906754778648553, -3.6635616461296463, 0.0,
        //   3.6635616461296463,  6.906754778648553]
        let p = vec64![0.001, 0.025, 0.5, 0.975, 0.999];
        let expect = [
            -6.906754778648553,
            -3.6635616461296463,
            0.0,
            3.6635616461296463,
            6.906754778648553,
        ];
        let arr = dense_data(logistic_quantile(&p, 0.0, 1.0, None, None).unwrap());
        for (a, e) in arr.iter().zip(expect.iter()) {
            assert_close(*a, *e, 3e-14);
        }
    }

    #[test]
    fn logistic_quantile_parametrised() {
        // Choose x = [1,3,5] with μ=1, s=2
        // p = logistic_cdf(x)
        let p = vec64![0.5, 0.7310585786300049, 0.8807970779778823];
        let expect = vec64![1.0, 3.0, 5.0];
        let arr = dense_data(logistic_quantile(&p, 1.0, 2.0, None, None).unwrap());
        for (a, e) in arr.iter().zip(expect.iter()) {
            assert_close(*a, *e, 2e-14);
        }
    }

    #[test]
    fn logistic_quantile_reflection_and_roundtrip() {
        // Reflection: Q(p) == -Q(1-p)  (μ=0, s=1)
        let p = vec64![1e-6, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1.0 - 1e-6];
        let q = dense_data(logistic_quantile(&p, 0.0, 1.0, None, None).unwrap());
        for (&pi, &qi) in p.iter().zip(q.iter()) {
            let q_reflect =
                dense_data(logistic_quantile(&[1.0 - pi], 0.0, 1.0, None, None).unwrap())[0];
            // Use larger tolerance for extreme values due to floating point precision
            let tol = if pi < 1e-5 || pi > 1.0 - 1e-5 {
                3e-11
            } else {
                2e-14
            };
            assert_close(qi, -q_reflect, tol);
        }

        // Round-trip  Q(F(x)) == x
        let x = vec64![-4.0, -1.0, 0.0, 1.5, 6.0];
        let cdf = dense_data(logistic_cdf(&x, 0.0, 1.0, None, None).unwrap());
        let ppf = dense_data(logistic_quantile(&cdf, 0.0, 1.0, None, None).unwrap());
        for (xi, ppi) in x.iter().zip(ppf.iter()) {
            // Use larger tolerance for extreme values
            let tol = if xi.abs() > 5.0 { 5e-14 } else { 2e-14 };
            assert_close(*xi, *ppi, tol);
        }
    }

    #[test]
    fn logistic_quantile_domain_and_mask() {
        let p = vec64![f64::NAN, -0.1, 0.0, 0.5, 1.0, 1.1];
        let mut mask = Bitmask::new_set_all(6, true);
        unsafe { mask.set_unchecked(2, false) }; // make entry 2 null
        let arr = logistic_quantile(&p, 0.0, 1.0, Some(&mask), Some(1)).unwrap();
        let nulls = mask_vec(arr.null_mask.as_ref().unwrap());

        // slots:      0     1      2      3     4      5
        // expected:  T     T      F      T     T      T
        assert_eq!(nulls, vec![true, true, false, true, true, true]);
        assert!(arr.data[0].is_nan());
        assert!(arr.data[1].is_nan());
        assert!(arr.data[2].is_nan());
        assert!(arr.data[3].is_finite());
        assert!(arr.data[4].is_infinite() && arr.data[4].is_sign_positive());
        assert!(arr.data[5].is_nan());
    }

    #[test]
    fn logistic_quantile_invalid_params() {
        assert!(logistic_quantile(&[0.5], 0.0, 0.0, None, None).is_err());
        assert!(logistic_quantile(&[0.5], f64::NAN, 1.0, None, None).is_err());
        assert!(logistic_quantile(&[0.5], 0.0, f64::INFINITY, None, None).is_err());
    }

    #[test]
    fn logistic_quantile_empty() {
        let arr = logistic_quantile(&[], 0.0, 1.0, None, None).unwrap();
        assert!(arr.data.is_empty());
        assert!(arr.null_mask.is_none());
    }
}
