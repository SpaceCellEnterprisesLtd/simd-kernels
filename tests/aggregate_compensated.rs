// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! Adversarial accuracy tests for Neumaier compensated summation.
//!
//! These tests use pathological input patterns that expose catastrophic
//! cancellation in naive floating-point summation. Compensated summation
//! should produce results within tight tolerances of known reference values.
//!
//! Note: Neumaier compensation is applied at the inter-chunk level, after
//! SIMD `reduce_sum()`. Intra-chunk error within a SIMD vector is limited
//! by the small number of SIMD lanes, which is acceptable. The adversarial
//! patterns here are designed to stress the inter-chunk accumulation path.

use minarrow::Vec64;
use simd_kernels::kernels::aggregate::{
    pair_stats_f64, stat_moments_f64, sum_f64_raw, sum_squares,
};

// ************************************
// Test 1: Large-small interleave across chunks
//
// Structure: blocks of large positive values, then blocks of large negative
// values, then blocks of small values. The large values cancel across chunks,
// leaving only the small values. Naive inter-chunk summation loses the small
// values. Neumaier compensation preserves them.
// ************************************
#[test]
fn test_compensated_sum_large_small_interleave() {
    let n = 1000;
    let mut data = Vec64::with_capacity(3 * n);

    // Block 1: n large positive values
    for _ in 0..n {
        data.push(1e15);
    }
    // Block 2: n large negative values (cancel block 1)
    for _ in 0..n {
        data.push(-1e15);
    }
    // Block 3: n small values that are the true answer
    for _ in 0..n {
        data.push(1.0);
    }

    let (result, count) = sum_f64_raw(&data, None, None);
    assert_eq!(count, 3 * n);

    // After cancellation, only the n * 1.0 terms should survive
    let expected = n as f64;
    assert!(
        (result - expected).abs() < 1e-6,
        "Large-small interleave: expected {expected}, got {result} (error = {})",
        (result - expected).abs()
    );
}

// ************************************
// Test 2: Alternating sign with small residual
//
// Pattern: [1e15, -1e15 + 1.0] repeated 10000 times.
// Each pair contributes 1.0, so expected sum = 10000.0.
// Naive summation loses the 1.0 when subtracting near-equal magnitudes.
// ************************************
#[test]
fn test_compensated_sum_alternating_sign() {
    let mut data = Vec64::with_capacity(20000);
    for _ in 0..10000 {
        data.push(1e15);
        data.push(-1e15 + 1.0);
    }

    let (result, count) = sum_f64_raw(&data, None, None);
    assert_eq!(count, 20000);
    assert!(
        (result - 10000.0).abs() < 1e-6,
        "Alternating sign: expected 10000.0, got {result} (error = {})",
        (result - 10000.0).abs()
    );
}

// ************************************
// Test 3: Monotone decreasing (harmonic series)
//
// Sum of 1/i for i in 1..=1_000_000 (harmonic number H_1000000).
// Known high-precision reference: H(1e6) ~ 14.392726722864989.
// This tests accumulation of many small terms where ordering matters.
// ************************************
#[test]
fn test_compensated_sum_harmonic() {
    let data: Vec64<f64> = (1..=1_000_000u64).map(|i| 1.0 / i as f64).collect();

    let (result, count) = sum_f64_raw(&data, None, None);
    assert_eq!(count, 1_000_000);

    let reference = 14.392726722864989;
    assert!(
        (result - reference).abs() < 1e-10,
        "Harmonic series: expected ~{reference}, got {result} (error = {})",
        (result - reference).abs()
    );
}

// ************************************
// Test 4: stat_moments compensation
//
// Dataset of [1e8 + 1.0, 1e8 + 2.0, ..., 1e8 + 100.0].
// Expected sum = 100 * 1e8 + (1 + 2 + ... + 100) = 1e10 + 5050 = 10000005050.0.
// The large constant offset challenges naive summation accuracy.
//
// For sum-of-squares, we compute via high-precision reference:
// each term = (1e8 + k)^2 = 1e16 + 2e8*k + k^2
// Σ = 100*1e16 + 2e8*5050 + 338350
// = 1e18 + 1010000000000 + 338350
// = 1000001010338350.0
// ************************************
#[test]
fn test_compensated_stat_moments() {
    let data: Vec64<f64> = (1..=100u64).map(|k| 1e8 + k as f64).collect();

    let (sum, sum2, count) = stat_moments_f64(&data, None, None);
    assert_eq!(count, 100);

    let expected_sum = 10_000_005_050.0;
    assert!(
        (sum - expected_sum).abs() < 1e-4,
        "stat_moments sum: expected {expected_sum}, got {sum} (error = {})",
        (sum - expected_sum).abs()
    );

    // Compute reference sum-of-squares with high precision
    // Σ(1e8 + k)² for k=1..100
    let expected_sum2: f64 = (1..=100u64)
        .map(|k| {
            let val = 1e8 + k as f64;
            val * val
        })
        .fold((0.0f64, 0.0f64), |(s, c), x| {
            let t = s + x;
            let new_c = if s.abs() >= x.abs() {
                c + (s - t) + x
            } else {
                c + (x - t) + s
            };
            (t, new_c)
        })
        .let_it(|(s, c)| s + c);

    let rel_err = ((sum2 - expected_sum2) / expected_sum2).abs();
    assert!(
        rel_err < 1e-10,
        "stat_moments sum2: expected {expected_sum2}, got {sum2} (relative error = {rel_err})"
    );
}

// Helper trait for pipeline-style let-binding
trait LetIt {
    fn let_it<R>(self, f: impl FnOnce(Self) -> R) -> R
    where
        Self: Sized;
}
impl<T> LetIt for T {
    fn let_it<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}

// ************************************
// Test 5: pair_stats compensation
//
// Two vectors with large offset. Verifies all 5 accumulators are accurate.
// x = [1e8 + 1, 1e8 + 2, ..., 1e8 + 100]
// y = [2e8 + 1, 2e8 + 2, ..., 2e8 + 100]
// ************************************
#[test]
fn test_compensated_pair_stats() {
    let xs: Vec64<f64> = (1..=100u64).map(|k| 1e8 + k as f64).collect();
    let ys: Vec64<f64> = (1..=100u64).map(|k| 2e8 + k as f64).collect();

    let ps = pair_stats_f64(&xs, &ys, None, None);
    assert_eq!(ps.n, 100);

    // sx = Σ(1e8 + k) = 1e10 + 5050
    let expected_sx = 10_000_005_050.0;
    assert!(
        (ps.sx - expected_sx).abs() < 1e-4,
        "pair_stats sx: expected {expected_sx}, got {} (error = {})",
        ps.sx,
        (ps.sx - expected_sx).abs()
    );

    // sy = Σ(2e8 + k) = 2e10 + 5050
    let expected_sy = 20_000_005_050.0;
    assert!(
        (ps.sy - expected_sy).abs() < 1e-4,
        "pair_stats sy: expected {expected_sy}, got {} (error = {})",
        ps.sy,
        (ps.sy - expected_sy).abs()
    );

    // Compute reference sxy with high precision using Neumaier fold
    let expected_sxy = neumaier_fold((1..=100u64).map(|k| {
        let x = 1e8 + k as f64;
        let y = 2e8 + k as f64;
        x * y
    }));
    let rel_err_sxy = ((ps.sxy - expected_sxy) / expected_sxy).abs();
    assert!(
        rel_err_sxy < 1e-10,
        "pair_stats sxy: expected {expected_sxy}, got {} (relative error = {rel_err_sxy})",
        ps.sxy
    );

    // Compute reference sx2 with high precision
    let expected_sx2 = neumaier_fold((1..=100u64).map(|k| {
        let x = 1e8 + k as f64;
        x * x
    }));
    let rel_err_sx2 = ((ps.sx2 - expected_sx2) / expected_sx2).abs();
    assert!(
        rel_err_sx2 < 1e-10,
        "pair_stats sx2: expected {expected_sx2}, got {} (relative error = {rel_err_sx2})",
        ps.sx2
    );

    // Compute reference sy2 with high precision
    let expected_sy2 = neumaier_fold((1..=100u64).map(|k| {
        let y = 2e8 + k as f64;
        y * y
    }));
    let rel_err_sy2 = ((ps.sy2 - expected_sy2) / expected_sy2).abs();
    assert!(
        rel_err_sy2 < 1e-10,
        "pair_stats sy2: expected {expected_sy2}, got {} (relative error = {rel_err_sy2})",
        ps.sy2
    );
}

/// Computes a Neumaier-compensated sum of an iterator, used as reference.
fn neumaier_fold(iter: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for x in iter {
        let t = sum + x;
        if t.is_finite() {
            if sum.abs() >= x.abs() {
                comp += (sum - t) + x;
            } else {
                comp += (x - t) + sum;
            }
        }
        sum = t;
    }
    sum + comp
}

// ************************************
// Test 6: sum_squares compensation
//
// Vector of [1e8 + k for k in 1..=1000].
// Reference computed via Neumaier fold to avoid manual arithmetic errors.
// ************************************
#[test]
fn test_compensated_sum_squares() {
    let data: Vec64<f64> = (1..=1000u64).map(|k| 1e8 + k as f64).collect();

    let result = sum_squares(&data, None, None);

    // Reference computed element-by-element with Neumaier compensation
    let expected = neumaier_fold((1..=1000u64).map(|k| {
        let v = 1e8 + k as f64;
        v * v
    }));

    let rel_err = ((result - expected) / expected).abs();
    assert!(
        rel_err < 1e-10,
        "sum_squares: expected {expected}, got {result} (relative error = {rel_err})"
    );
}
