// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Fast Fourier Transform Module** - *High-Performance Frequency Domain Analysis*
//!
//! This module implements optimised Fast Fourier Transform (FFT) algorithms for efficient
//! frequency domain analysis and signal processing applications. It provides both small-scale
//! radix-optimised transforms and large-scale blocked implementations for scientific
//! computing and digital signal processing workflows.
//!
//! ## Use cases
//!
//! The Fast Fourier Transform is fundamental to numerous computational domains:
//! - **Digital Signal Processing**: Spectral analysis, filtering, and convolution
//! - **Image Processing**: Frequency domain transformations and enhancement
//! - **Scientific Computing**: Numerical solution of PDEs via spectral methods
//! - **Audio Processing**: Frequency analysis and synthesis
//! - **Telecommunications**: Modulation, demodulation, and channel analysis
//! - **Machine Learning**: Feature extraction and data preprocessing

use minarrow::enums::error::KernelError;
use minarrow::{FloatArray, Vec64};
use num_complex::Complex64;

#[inline(always)]
pub fn butterfly_radix8(buf: &mut [Complex64]) {
    debug_assert_eq!(buf.len(), 8);

    // Split into even/odd halves and reuse temporaries to minimise loads
    let (x0, x1, x2, x3, x4, x5, x6, x7) = (
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    );

    // First layer (radix-2 butterflies)
    let a04 = x0 + x4;
    let s04 = x0 - x4;
    let a26 = x2 + x6;
    let s26 = x2 - x6;
    let a15 = x1 + x5;
    let s15 = x1 - x5;
    let a37 = x3 + x7;
    let s37 = x3 - x7;

    // Second layer (radix-4)
    let a04a26 = a04 + a26;
    let a04s26 = a04 - a26;
    let a15a37 = a15 + a37;
    let a15s37 = a15 - a37;

    // Calculate ±i·(something) once
    const J: Complex64 = Complex64 { re: 0.0, im: 1.0 };

    // Radix-8 output
    buf[0] = a04a26 + a15a37;
    buf[4] = a04a26 - a15a37;

    let t0 = s04 + J * s26;
    let t1 = s15 + J * s37;
    buf[2] = t0 + Complex64::new(0.0, -1.0) * t1; //  e^{-jπ/2}
    buf[6] = t0 + Complex64::new(0.0, 1.0) * t1; //  e^{ jπ/2}

    let u0 = a04s26;
    let u1 = Complex64::new(0.0, -1.0) * a15s37;
    buf[1] = u0 + u1; //  e^{-jπ/4} merged factor
    buf[5] = u0 - u1; //  e^{ 3jπ/4}

    let v0 = s04 - J * s26;
    let v1 = s15 - J * s37;
    buf[3] = v0 - Complex64::new(0.0, 1.0) * v1; //  e^{ jπ/2}
    buf[7] = v0 - Complex64::new(0.0, -1.0) * v1; //  e^{-jπ/2}
}

// In-place radix-4 DIT for 4 points.
#[inline(always)]
fn fft4_in_place(x: &mut [Complex64; 4]) {
    let x0 = x[0];
    let x1 = x[1];
    let x2 = x[2];
    let x3 = x[3];

    let a = x0 + x2; // (0)+(2)
    let b = x0 - x2; // (0)-(2)
    let c = x1 + x3; // (1)+(3)
    let d = (x1 - x3) * Complex64::new(0.0, -1.0); // (1)-(3) times -j

    x[0] = a + c; // k=0
    x[2] = a - c; // k=2
    x[1] = b + d; // k=1
    x[3] = b - d; // k=3
}

/// 8-point FFT
/// 8-point FFT (radix-2/4 DIT): split evens/odds -> FFT4 each -> twiddle & combine.
#[inline(always)]
pub fn fft8_radix(
    buf: &mut [Complex64; 8],
) -> Result<(FloatArray<f64>, FloatArray<f64>), KernelError> {
    // Split into evens and odds
    let mut even = [buf[0], buf[2], buf[4], buf[6]];
    let mut odd = [buf[1], buf[3], buf[5], buf[7]];

    // 4-point FFTs
    fft4_in_place(&mut even);
    fft4_in_place(&mut odd);

    // Twiddles W8^k = exp(-j*2π*k/8)
    // W8^0 = 1 + 0j
    // W8^1 =  √2/2 - j√2/2
    // W8^2 =  0   - j
    // W8^3 = -√2/2 - j√2/2
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let w1 = Complex64::new(s, -s);
    let w2 = Complex64::new(0.0, -1.0);
    let w3 = Complex64::new(-s, -s);

    let t0 = odd[0]; // W8^0 * odd[0]
    let t1 = w1 * odd[1]; // W8^1 * odd[1]
    let t2 = w2 * odd[2]; // W8^2 * odd[2]
    let t3 = w3 * odd[3]; // W8^3 * odd[3]

    buf[0] = even[0] + t0;
    buf[4] = even[0] - t0;

    buf[1] = even[1] + t1;
    buf[5] = even[1] - t1;

    buf[2] = even[2] + t2;
    buf[6] = even[2] - t2;

    buf[3] = even[3] + t3;
    buf[7] = even[3] - t3;

    // Package outputs (same as your function did)
    let mut real = Vec64::with_capacity(8);
    let mut imag = Vec64::with_capacity(8);
    for &z in buf.iter() {
        real.push(z.re);
        imag.push(z.im);
    }
    Ok((FloatArray::new(real, None), FloatArray::new(imag, None)))
}

/// Power-of-two, in-place FFT (≥8, radix-2 stages, radix-8 leaf).
#[inline]
pub fn block_fft(
    data: &mut [Complex64],
) -> Result<(FloatArray<f64>, FloatArray<f64>), KernelError> {
    let n = data.len();
    if n < 2 || (n & (n - 1)) != 0 {
        return Err(KernelError::InvalidArguments(
            "block_fft: N must be power-of-two and ≥2".into(),
        ));
    }

    // bit-reversal permutation
    let bits = n.trailing_zeros();
    for i in 0..n {
        let rev = i.reverse_bits() >> (usize::BITS - bits);
        if i < rev {
            data.swap(i, rev);
        }
    }

    // iterative radix-2 DIT
    let mut m = 2;
    while m <= n {
        let half = m / 2;
        let theta = -2.0 * std::f64::consts::PI / (m as f64);
        let w_m = Complex64::from_polar(1.0, theta);

        for k in (0..n).step_by(m) {
            let mut w = Complex64::new(1.0, 0.0);
            for j in 0..half {
                let t = w * data[k + j + half];
                let u = data[k + j];
                data[k + j] = u + t;
                data[k + j + half] = u - t;
                w *= w_m;
            }
        }
        m <<= 1;
    }

    let mut real = Vec64::with_capacity(n);
    let mut imag = Vec64::with_capacity(n);
    for &z in data.iter() {
        real.push(z.re);
        imag.push(z.im);
    }
    Ok((FloatArray::new(real, None), FloatArray::new(imag, None)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use rand::Rng;

    // ---- SciPy/NumPy FFT references ----

    fn scipy_fft_ref_8_seq_0_7() -> [Complex64; 8] {
        [
            Complex64::new(28.0, 0.0),
            Complex64::new(-4.0, 9.6568542494923797),
            Complex64::new(-4.0, 4.0),
            Complex64::new(-4.0, 1.6568542494923806),
            Complex64::new(-4.0, 0.0),
            Complex64::new(-4.0, -1.6568542494923806),
            Complex64::new(-4.0, -4.0),
            Complex64::new(-4.0, -9.6568542494923797),
        ]
    }

    fn scipy_fft_ref_16_seq_0_15() -> [Complex64; 16] {
        [
            Complex64::new(120.0, 0.0),
            Complex64::new(-7.9999999999999991, 40.218715937006785),
            Complex64::new(-8.0, 19.313708498984759),
            Complex64::new(-7.9999999999999991, 11.972846101323913),
            Complex64::new(-8.0, 8.0),
            Complex64::new(-8.0, 5.345429103354391),
            Complex64::new(-8.0, 3.3137084989847612),
            Complex64::new(-8.0, 1.5912989390372658),
            Complex64::new(-8.0, 0.0),
            Complex64::new(-7.9999999999999991, -1.5912989390372658),
            Complex64::new(-8.0, -3.3137084989847612),
            Complex64::new(-7.9999999999999991, -5.3454291033543946),
            Complex64::new(-8.0, -8.0),
            Complex64::new(-8.0, -11.97284610132391),
            Complex64::new(-8.0, -19.313708498984759),
            Complex64::new(-8.0, -40.218715937006785),
        ]
    }

    #[test]
    fn butterfly_radix8_impulse_all_ones() {
        let mut buf = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        butterfly_radix8(&mut buf);
        let ones = [Complex64::new(1.0, 0.0); 8];
        assert_vec_close(&buf, &ones, 1e-15);
    }

    #[test]
    fn fft8_radix_matches_scipy_seq0_7() {
        let mut buf = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(7.0, 0.0),
        ];
        let (_re, _im) = fft8_radix(&mut buf).unwrap();
        let ref_out = scipy_fft_ref_8_seq_0_7();
        assert_vec_close(&buf, &ref_out, 1e-12);
    }

    #[test]
    fn block_fft_matches_scipy_seq0_7() {
        let mut data = (0..8)
            .map(|v| Complex64::new(v as f64, 0.0))
            .collect::<Vec<_>>();
        let (_re, _im) = block_fft(&mut data).unwrap();
        let ref_out = scipy_fft_ref_8_seq_0_7();
        assert_vec_close(&data, &ref_out, 1e-12);
    }

    #[test]
    fn block_fft_matches_scipy_seq0_15() {
        let mut data = (0..16)
            .map(|v| Complex64::new(v as f64, 0.0))
            .collect::<Vec<_>>();
        let (_re, _im) = block_fft(&mut data).unwrap();
        let ref_out = scipy_fft_ref_16_seq_0_15();
        assert_vec_close(&data, &ref_out, 1e-11);
    }

    // Basic DFT for validation
    fn dft_naive(x: &[Complex64]) -> Vec<Complex64> {
        let n = x.len() as f64;
        (0..x.len())
            .map(|k| {
                let mut sum = Complex64::new(0.0, 0.0);
                for (n_idx, &val) in x.iter().enumerate() {
                    let angle = -2.0 * std::f64::consts::PI * (k as f64) * (n_idx as f64) / n;
                    sum += val * Complex64::from_polar(1.0, angle);
                }
                sum
            })
            .collect()
    }

    fn assert_vec_close(a: &[Complex64], b: &[Complex64], eps: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).norm() < eps, "mismatch: x={:?}, y={:?}", x, y);
        }
    }

    #[test]
    fn radix8_exact() {
        let mut buf = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(7.0, 0.0),
        ];
        let (_, _) = fft8_radix(&mut buf).unwrap();
        let ref_out = dft_naive(&[
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(7.0, 0.0),
        ]);
        assert_vec_close(&buf, &ref_out, 1e-12);
    }

    #[test]
    fn block_fft_random_lengths() {
        let mut rng = rand::rng();
        for &n in &[8, 16, 32, 64, 128, 256, 512, 1024] {
            let mut data: Vec<Complex64> = (0..n)
                .map(|_| Complex64::new(rng.random(), rng.random()))
                .collect();
            let ref_data = data.clone();
            let (_, _) = block_fft(&mut data).unwrap();
            let ref_out = dft_naive(&ref_data);
            assert_vec_close(&data, &ref_out, 1e-9); // generous for large n
        }
    }

    #[test]
    fn block_fft_power_of_two_check() {
        let mut bad = vec![Complex64::new(0.0, 0.0); 12]; // not power of two
        assert!(block_fft(&mut bad).is_err());
    }
}
