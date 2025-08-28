// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Chi-Squared Distribution** - *Critical Values and Goodness-of-Fit Testing*
//!
//! High-performance implementation of the chi-squared distribution providing probability
//! density functions, cumulative distribution functions, and quantile calculations
//! with SIMD acceleration and numerical precision guarantees.
//!
//! ### Parameters
//! - **`df` (degrees of freedom)**: Shape parameter `k > 0`
//!
//! ### Moment Properties
//! - **Mean**: `k`
//! - **Variance**: `2k`
//! - **Skewness**: `sqrt(8/k)`
//! - **Support**: `[0, ∞)`
//!
//! ## Applications
//! - **Hypothesis testing**: Chi-squared goodness-of-fit and independence tests
//! - **Confidence intervals**: For variance estimates in normal populations
//! - **Model selection**: Likelihood ratio test statistics
//! - **Quality control**: Process variation analysis
#[cfg(feature = "simd")]
mod simd;
mod std;

use minarrow::Bitmask;
use minarrow::FloatArray;
use minarrow::enums::error::KernelError;

/// Chi-square PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn chi_square_pdf_to(
    x: &[f64],
    df: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::chi_square_pdf_simd_to(x, df, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::chi_square_pdf_std_to(x, df, output, null_mask, null_count)
    }
}

/// Chi-square PDF: f(x; k) = 1/(2^{k/2} Γ(k/2)) x^{k/2-1} e^{-x/2}
#[inline(always)]
pub fn chi_square_pdf(
    x: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::chi_square_pdf_simd(x, df, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::chi_square_pdf_std(x, df, null_mask, null_count)
    }
}

/// Chi-square CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn chi_square_cdf_to(
    x: &[f64],
    df: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::chi_square_cdf_std_to(x, df, output, null_mask, null_count)
}

/// Chi-square CDF: F(x; k) = P(k/2, x/2)
#[inline(always)]
pub fn chi_square_cdf(
    x: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::chi_square_cdf_std(x, df, null_mask, null_count)
}

/// Chi-square quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn chi_square_quantile_to(
    p: &[f64],
    df: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::chi_square_quantile_std_to(p, df, output, null_mask, null_count)
}

/// Chi-square quantile function (inverse CDF).
///
/// For each `p` in `p`, returns the value `x` such that `P(X ≤ x) = p` for
/// a chi-square distribution with `df` degrees of freedom.
#[inline(always)]
pub fn chi_square_quantile(
    p: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::chi_square_quantile_std(p, df, null_mask, null_count)
}

#[cfg(test)]
mod chi_square_tests {
    use ::std::f64::consts::LN_2;

    use crate::kernels::scientific::distributions::{
        shared::scalar::{inv_reg_lower_gamma, ln_gamma},
        univariate::common::dense_data,
    };

    use super::*;
    use minarrow::{Bitmask, vec64};

    // helpers

    fn mask_vec(m: &Bitmask) -> Vec<bool> {
        (0..m.len()).map(|i| m.get(i)).collect()
    }
    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "assert_close failed: {a} vs {b} (tol={tol})"
        );
    }

    // Reference cases used during early testing without the library / SIMD code
    // Later scipy test suite was implemented under "./tests"
    fn pdf_ref(x: f64, k: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let k2 = 0.5 * k;
        let log_norm = -k2 * LN_2 - ln_gamma(k2);
        (log_norm + (k2 - 1.0) * x.ln() - 0.5 * x).exp()
    }

    fn quantile_ref(p: f64, k: f64) -> f64 {
        if p == 0.0 {
            0.0
        } else if p == 1.0 {
            f64::INFINITY
        } else {
            2.0 * inv_reg_lower_gamma(0.5 * k, p)
        }
    }

    // PDF tests
    #[test]
    fn chi_square_pdf_basic() {
        let xs = vec64![0.5, 1.0, 2.0, 5.0];
        let df = 3.0;
        let expect: Vec<f64> = xs.iter().map(|&x| pdf_ref(x, df)).collect();
        let out = dense_data(chi_square_pdf(&xs, df, None, None).unwrap());
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-14);
        }
    }

    #[test]
    fn chi_square_pdf_bulk_vs_scalar() {
        let xs = vec64![0.1, 0.9, 1.5, 4.3];
        let bulk = dense_data(chi_square_pdf(&xs, 5.0, None, None).unwrap());
        for (i, &x) in xs.iter().enumerate() {
            let x_aligned = vec64![x];
            let scalar =
                dense_data(chi_square_pdf(x_aligned.as_slice(), 5.0, None, None).unwrap())[0];
            assert_close(bulk[i], scalar, 1e-15);
        }
    }

    #[test]
    fn chi_square_pdf_negative_and_infinite_inputs() {
        let xs = vec64![-1.0, f64::INFINITY];
        let out = dense_data(chi_square_pdf(&xs, 4.0, None, None).unwrap());
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn chi_square_pdf_mask_propagation() {
        let xs = vec64![0.5, 1.0, 2.0];
        let mut m = Bitmask::new_set_all(3, true);
        unsafe {
            m.set_unchecked(1, false);
        }
        let arr = chi_square_pdf(&xs, 2.0, Some(&m), Some(1)).unwrap();
        assert_eq!(
            mask_vec(arr.null_mask.as_ref().unwrap()),
            vec![true, false, true]
        );
        assert!(arr.data[1].is_nan());
    }

    // CDF tests
    #[test]
    fn chi_square_cdf_reference() {
        let xs = vec64![0.1, 1.0, 3.0, 10.0];
        let df = 2.5;
        // SciPy truth values: stats.chi2.cdf([0.1, 1.0, 3.0, 10.0], 2.5)
        let exp = vec64![
            0.020298266579604166,
            0.28378995266531293,
            0.6941503705541809,
            0.9883919628521234
        ];
        let out = dense_data(chi_square_cdf(&xs, df, None, None).unwrap());
        for (a, e) in out.iter().zip(exp.iter()) {
            assert_close(*a, *e, 1e-14);
        }
    }

    #[test]
    fn chi_square_cdf_tail_limits() {
        let xs = vec64![-1e6, 1e6];
        let out = dense_data(chi_square_cdf(&xs, 5.0, None, None).unwrap());
        assert_close(out[0], 0.0, 1e-15);
        assert_close(out[1], 1.0, 1e-15);
    }

    #[test]
    fn chi_square_cdf_mask_nan() {
        let xs = vec64![0.0, f64::NAN];
        let mut m = Bitmask::new_set_all(2, true);
        unsafe {
            m.set_unchecked(1, false);
        }
        let arr = chi_square_cdf(&xs, 3.0, Some(&m), Some(1)).unwrap();
        let nulls = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(nulls, vec![true, false]);
        assert!(arr.data[1].is_nan());
    }

    // Quantile tests
    #[test]
    fn chi_square_quantile_reference_values() {
        let probs = vec64![0.001, 0.1, 0.5, 0.9, 0.999];
        let df = 4.0;
        let exp: Vec<f64> = probs.iter().map(|&p| quantile_ref(p, df)).collect();
        let out = dense_data(chi_square_quantile(&probs, df, None, None).unwrap());
        for (a, e) in out.iter().zip(exp.iter()) {
            assert_close(*a, *e, 1e-11);
        }
    }

    #[test]
    fn chi_square_quantile_round_trip() {
        let xs = vec64![0.5, 1.3, 4.0, 9.0];
        let df = 7.0;
        let ps = dense_data(chi_square_cdf(&xs, df, None, None).unwrap());
        let back = dense_data(chi_square_quantile(&ps, df, None, None).unwrap());
        for (x, b) in xs.iter().zip(back.iter()) {
            assert_close(*x, *b, 1e-11);
        }
    }

    #[test]
    fn chi_square_quantile_domain_edges() {
        let ps = vec64![0.0, 1.0, -0.1, 1.1, f64::NAN];
        let out = dense_data(chi_square_quantile(&ps, 3.0, None, None).unwrap());
        assert_eq!(out[0], 0.0);
        assert!(out[1].is_infinite());
        assert!(out[2].is_nan() && out[3].is_nan() && out[4].is_nan());
    }

    #[test]
    fn chi_square_quantile_mask() {
        let ps = vec64![0.2, 0.8];
        let mut m = Bitmask::new_set_all(2, true);
        unsafe {
            m.set_unchecked(0, false);
        }
        let arr = chi_square_quantile(&ps, 5.0, Some(&m), Some(1)).unwrap();
        let nulls = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(nulls, vec![false, true]);
        assert!(arr.data[0].is_nan());
    }

    // Parameter / input validation
    #[test]
    fn chi_square_invalid_params() {
        assert!(chi_square_pdf(&[1.0], 0.0, None, None).is_err());
        assert!(chi_square_cdf(&[1.0], -2.0, None, None).is_err());
        assert!(chi_square_quantile(&[0.5], f64::NAN, None, None).is_err());
    }

    #[test]
    fn chi_square_empty_inputs() {
        assert!(
            chi_square_pdf(&[], 1.0, None, None)
                .unwrap()
                .data
                .is_empty()
        );
        assert!(
            chi_square_cdf(&[], 1.0, None, None)
                .unwrap()
                .data
                .is_empty()
        );
        assert!(
            chi_square_quantile(&[], 1.0, None, None)
                .unwrap()
                .data
                .is_empty()
        );
    }
}
