// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Student's t-Distribution SIMD Implementations** - *Vectorised Heavy-Tailed Statistical Inference*
//!
//! High-performance SIMD-accelerated implementations of Student's t-distribution pdf function.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::{Simd, StdFloat};

use minarrow::utils::is_simd_aligned;
use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::shared::scalar::*;
use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_f64_simd_to, masked_univariate_kernel_f64_simd_to,
};
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std_to, masked_univariate_kernel_f64_std_to,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// SIMD-accelerated Student's t-distribution PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn student_t_pdf_simd_to(
    x: &[f64],
    df: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if df <= 0.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "student_t_pdf: invalid df".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    const N: usize = W64;
    let coeff =
        (ln_gamma((df + 1.0) * 0.5) - ln_gamma(df * 0.5) - 0.5 * (df * std::f64::consts::PI).ln())
            .exp();

    let scalar_body =
        move |xi: f64| -> f64 { coeff * (1.0 + xi * xi / df).powf(-(df + 1.0) * 0.5) };

    let simd_body = move |x_v: Simd<f64, N>| -> Simd<f64, N> {
        let df_v = Simd::<f64, N>::splat(df);
        let coeff_v = Simd::<f64, N>::splat(coeff);
        let exp_v = Simd::<f64, N>::splat(-(df + 1.0) * 0.5);
        let one_v = Simd::<f64, N>::splat(1.0);
        let t = one_v + (x_v * x_v) / df_v;
        coeff_v * (t.ln() * exp_v).exp()
    };

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(x) {
            dense_univariate_kernel_f64_simd_to::<N, _, _>(x, output, simd_body, scalar_body);
            return Ok(());
        }
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("student_t_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    if is_simd_aligned(x) {
        masked_univariate_kernel_f64_simd_to::<N, _, _>(
            x,
            mask_ref,
            output,
            &mut out_mask,
            simd_body,
            scalar_body,
        );
        return Ok(());
    }
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);
    Ok(())
}

/// **Student's t-Distribution Probability Density Function** - *SIMD-Accelerated Heavy-Tailed PDF*
///
/// Computes the probability density function of Student's t-distribution using vectorised SIMD operations
/// where possible, with automatic scalar fallback for compatibility across all architectures.
#[inline(always)]
pub fn student_t_pdf_simd(
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

    student_t_pdf_simd_to(x, df, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
