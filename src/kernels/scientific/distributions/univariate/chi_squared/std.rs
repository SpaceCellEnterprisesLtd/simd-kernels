// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::shared::scalar::*;
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std_to, masked_univariate_kernel_f64_std_to,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// Chi-square PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn chi_square_pdf_std_to(
    x: &[f64],
    df: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if df <= 0.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "chi_square_pdf: invalid df".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let k2 = 0.5 * df;
    let log_norm = -k2 * std::f64::consts::LN_2 - ln_gamma(k2);

    let scalar_body = move |xi: f64| {
        if xi < 0.0 {
            0.0
        } else if xi == 0.0 && k2 < 1.0 {
            f64::INFINITY
        } else if xi == 0.0 && k2 == 1.0 {
            log_norm.exp()
        } else if xi == 0.0 {
            0.0
        } else if xi.is_infinite() {
            0.0
        } else if xi.is_nan() {
            f64::NAN
        } else {
            (log_norm + (k2 - 1.0) * xi.ln() - 0.5 * xi).exp()
        }
    };

    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("chi_square_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);
    Ok(())
}

/// Chi-square PDF: f(x; k) = 1/(2^{k/2} Γ(k/2)) x^{k/2-1} e^{-x/2}
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn chi_square_pdf_std(
    x: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    chi_square_pdf_std_to(x, df, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Chi-square CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn chi_square_cdf_std_to(
    x: &[f64],
    df: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if df <= 0.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "chi_square_cdf: invalid df".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let len = x.len();
    let k2 = 0.5 * df;

    let eval = |xi: f64| -> f64 {
        if xi.is_nan() {
            f64::NAN
        } else if xi < 0.0 {
            0.0
        } else if xi.is_infinite() {
            1.0
        } else {
            reg_lower_gamma(k2, 0.5 * xi)
        }
    };

    if !has_nulls(null_count, null_mask) {
        for (i, &xi) in x.iter().enumerate() {
            output[i] = eval(xi);
        }
        return Ok(());
    }

    let mask = null_mask.expect("null path requires a mask");
    for i in 0..len {
        if !unsafe { mask.get_unchecked(i) } {
            output[i] = f64::NAN;
        } else {
            output[i] = eval(x[i]);
        }
    }
    Ok(())
}

/// Chi-square CDF: F(x; k) = P(k/2, x/2)
#[inline(always)]
pub fn chi_square_cdf_std(
    x: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    chi_square_cdf_std_to(x, df, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Chi-square quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
pub fn chi_square_quantile_std_to(
    p: &[f64],
    df: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if df <= 0.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "chi_square_quantile: invalid df".into(),
        ));
    }
    if p.is_empty() {
        return Ok(());
    }

    let a = 0.5 * df;
    let ln_gamma_ap1 = ln_gamma(a + 1.0);
    const SMALL_P: f64 = 1e-200;
    let len = p.len();

    let solve_one = |prob: f64| -> f64 {
        if !(prob >= 0.0 && prob <= 1.0) || !prob.is_finite() {
            return f64::NAN;
        }
        if prob == 0.0 {
            return 0.0;
        }
        if prob == 1.0 {
            return f64::INFINITY;
        }
        if prob < SMALL_P {
            let ln_x_half = (prob.ln() + ln_gamma_ap1) / a;
            return 2.0 * ln_x_half.exp();
        }

        let mut x0 = if df == 1.0 && prob < 0.1 {
            let z = inv_std_normal(0.5 * (1.0 + prob));
            (z * z).max(1e-100)
        } else {
            let nu = df;
            let z = inv_std_normal(prob);
            let c = 2.0 / (9.0 * nu);
            let base = 1.0 - (2.0 / (9.0 * nu)) + z * c.sqrt();
            (nu * base * base * base).max(0.0)
        };

        if !x0.is_finite() || x0 == 0.0 {
            x0 = 2.0 * inv_reg_lower_gamma(a, prob).max(1e-100);
        }

        if prob > 0.999 {
            chi2_newton_refine_extreme(x0, a, prob)
        } else {
            chi2_newton_refine(x0, a, prob)
        }
    };

    if !has_nulls(null_count, null_mask) {
        for (i, &prob) in p.iter().enumerate() {
            output[i] = solve_one(prob);
        }
        return Ok(());
    }

    let mask = null_mask.expect("null path requires a mask");
    for i in 0..len {
        if !unsafe { mask.get_unchecked(i) } {
            output[i] = f64::NAN;
        } else {
            output[i] = solve_one(p[i]);
        }
    }
    Ok(())
}

/// Chi-square quantile function (inverse CDF).
///
/// For each `p` in `p`, returns the value `x` such that `P(X ≤ x) = p` for
/// a chi-square distribution with `df` degrees of freedom.
pub fn chi_square_quantile_std(
    p: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    chi_square_quantile_std_to(p, df, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
