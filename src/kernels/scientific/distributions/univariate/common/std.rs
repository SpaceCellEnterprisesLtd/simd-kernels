// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

use minarrow::{Bitmask, Vec64};

/// Dense kernel helper (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
///
/// ### Null handling
/// - Same semantics as `dense_univariate_kernel_f64_std` but without allocation.
/// - Any `NaN` or `inf` values generated in the kernel function are kept verbatim.
///
/// # Panics
/// Panics if `x.len() != out.len()`.
#[inline(always)]
pub fn dense_univariate_kernel_f64_std_to<FScalar>(x: &[f64], out: &mut [f64], scalar_body: FScalar)
where
    FScalar: Fn(f64) -> f64,
{
    assert_eq!(
        x.len(),
        out.len(),
        "dense_univariate_kernel_f64_std_to: input/output length mismatch"
    );

    for (i, &xi) in x.iter().enumerate() {
        out[i] = scalar_body(xi);
    }
}

/// Dense kernel helper
///
/// ### Null handling
/// - Null mask appearing in the dense path means a mask was supplied
/// to the kernel function, with a null_count of `0`. This can reflect
/// a scenario where one knew there was no nulls for a whole vector, or
/// the supplied window, and therefore supplied `0` to ensure that the
/// dense path was used for the kernel.
/// - Any `NaN` or `inf` values generated in the kernel function
/// are kept verbatim, without `nulling` them in the (optional)
/// mask, given that:
/// 1. These values can represent additional data signal.
/// 2. Handling them requires additional CPU cycles on the hot path.
///
/// Therefore, one can treat them further if needed.
#[inline(always)]
pub fn dense_univariate_kernel_f64_std<FScalar>(
    x: &[f64],
    has_mask: bool,
    scalar_body: FScalar,
) -> (Vec64<f64>, Option<Bitmask>)
where
    FScalar: Fn(f64) -> f64,
{
    let len = x.len();
    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    dense_univariate_kernel_f64_std_to(x, out.as_mut_slice(), scalar_body);

    let out_mask = if has_mask {
        Some(Bitmask::new_set_all(len, true))
    } else {
        None
    };
    (out, out_mask)
}

/// Null-aware masked kernel helper (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer and mask.
///
/// ### Null handling
/// - Input mask is required and propagates nulls accordingly.
/// - Any `NaN` or `inf` values generated are kept verbatim.
///
/// # Panics
/// Panics if `x.len() != out.len()`.
#[inline(always)]
pub fn masked_univariate_kernel_f64_std_to<FScalar>(
    x: &[f64],
    mask: &Bitmask,
    out: &mut [f64],
    out_mask: &mut Bitmask,
    scalar_body: FScalar,
) where
    FScalar: Fn(f64) -> f64,
{
    assert_eq!(
        x.len(),
        out.len(),
        "masked_univariate_kernel_f64_std_to: input/output length mismatch"
    );

    process_scalar_masked_f64_std_to(x, mask, 0, out, out_mask, &scalar_body);
}

/// Null-aware masked kernel helper.
///
/// ### Null handling
/// - Input mask is required and propagates nulls accordingly.
/// - Any `NaN` or `inf` values generated in the kernel function
/// are kept verbatim, without `nulling` them in the (optional)
/// mask, given that:
/// 1. These values can represent additional data signal.
/// 2. Handling them requires additional CPU cycles on the hot path.
/// Therefore, one can treat them further if desired.
#[inline(always)]
pub fn masked_univariate_kernel_f64_std<FScalar>(
    x: &[f64],
    mask: &Bitmask,
    scalar_body: FScalar,
) -> (Vec64<f64>, Bitmask)
where
    FScalar: Fn(f64) -> f64,
{
    let len = x.len();
    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };
    let mut out_mask = mask.clone();

    masked_univariate_kernel_f64_std_to(x, mask, out.as_mut_slice(), &mut out_mask, scalar_body);

    (out, out_mask)
}

/// Processes the scalar section for masked kernels (zero-allocation variant).
///
/// Writes directly to caller-provided output slice.
///
/// # Safety
/// Uses unchecked access for performance within validated ranges.
#[inline(always)]
pub fn process_scalar_masked_f64_std_to<FScalar>(
    x: &[f64],
    mask: &Bitmask,
    start: usize,
    out: &mut [f64],
    out_mask: &mut Bitmask,
    scalar_body: &FScalar,
) where
    FScalar: Fn(f64) -> f64,
{
    let len = x.len();
    for idx in start..len {
        if !unsafe { mask.get_unchecked(idx) } {
            out[idx] = f64::NAN;
            unsafe { out_mask.set_unchecked(idx, false) };
        } else {
            let xi = unsafe { *x.get_unchecked(idx) };
            out[idx] = scalar_body(xi);
            unsafe { out_mask.set_unchecked(idx, true) };
        }
    }
}

/// Processes the scalar section for masked kernels.
///
/// # Safety
/// Uses unchecked access for performance within validated ranges.
#[inline(always)]
pub fn process_scalar_masked_f64_std<FScalar>(
    x: &[f64],
    mask: &Bitmask,
    start: usize,
    out: &mut Vec64<f64>,
    out_mask: &mut Bitmask,
    scalar_body: &FScalar,
) where
    FScalar: Fn(f64) -> f64,
{
    let len = x.len();
    for idx in start..len {
        if !unsafe { mask.get_unchecked(idx) } {
            out.push(f64::NAN);
            unsafe { out_mask.set_unchecked(idx, false) };
        } else {
            let xi = unsafe { *x.get_unchecked(idx) };
            out.push(scalar_body(xi));
            unsafe { out_mask.set_unchecked(idx, true) };
        }
    }
}

/// Dense kernel helper for u64->f64 kernels (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
///
/// # Parameters
/// - `x`: input `u64` data
/// - `out`: output buffer (must match input length)
/// - `scalar_body`: given each scalar `u64`, produce its `f64` result
///
/// # Panics
/// Panics if `x.len() != out.len()`.
#[inline(always)]
pub fn dense_univariate_kernel_u64_std_to<FScalar>(x: &[u64], out: &mut [f64], scalar_body: FScalar)
where
    FScalar: Fn(u64) -> f64,
{
    assert_eq!(
        x.len(),
        out.len(),
        "dense_univariate_kernel_u64_std_to: input/output length mismatch"
    );

    for (i, &ki) in x.iter().enumerate() {
        out[i] = scalar_body(ki);
    }
}

/// Dense kernel helper for u64->f64 kernels.
///
/// # Parameters
/// - `x`:      input `u64` data
/// - `has_mask`: whether to return an "all‐true" mask
/// - `scalar_body`: given each scalar `u64`, produce its `f64` result
#[inline(always)]
pub fn dense_univariate_kernel_u64_std<FScalar>(
    x: &[u64],
    has_mask: bool,
    scalar_body: FScalar,
) -> (Vec64<f64>, Option<Bitmask>)
where
    FScalar: Fn(u64) -> f64,
{
    let len = x.len();
    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    dense_univariate_kernel_u64_std_to(x, out.as_mut_slice(), scalar_body);

    let out_mask = if has_mask {
        Some(Bitmask::new_set_all(len, true))
    } else {
        None
    };
    (out, out_mask)
}

/// Null‐aware masked kernel helper for u64->f64 kernels (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer and mask.
///
/// # Panics
/// Panics if `x.len() != out.len()`.
#[inline(always)]
pub fn masked_univariate_kernel_u64_std_to<FScalar>(
    x: &[u64],
    mask: &Bitmask,
    out: &mut [f64],
    out_mask: &mut Bitmask,
    scalar_body: FScalar,
) where
    FScalar: Fn(u64) -> f64,
{
    assert_eq!(
        x.len(),
        out.len(),
        "masked_univariate_kernel_u64_std_to: input/output length mismatch"
    );

    process_scalar_masked_u64_std_to(x, mask, 0, out, out_mask, &scalar_body);
}

/// Null‐aware masked kernel helper for u64->f64 kernels.
///
/// Propagates the input mask and leaves any "invalid" (null) lanes as NULL;
/// lanes that are in the mask but produce a non‐finite result in the scalar
/// or SIMD body also become NULL.
#[inline(always)]
pub fn masked_univariate_kernel_u64_std<FScalar>(
    x: &[u64],
    mask: &Bitmask,
    scalar_body: FScalar,
) -> (Vec64<f64>, Bitmask)
where
    FScalar: Fn(u64) -> f64,
{
    let len = x.len();
    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };
    let mut out_mask = mask.clone();

    masked_univariate_kernel_u64_std_to(x, mask, out.as_mut_slice(), &mut out_mask, scalar_body);

    (out, out_mask)
}

/// Processes the scalar section for masked `u64->f64` kernels (zero-allocation variant).
///
/// Writes directly to caller-provided output slice.
///
/// # Safety
/// Uses unchecked access for performance within validated ranges.
#[inline(always)]
pub fn process_scalar_masked_u64_std_to<FScalar>(
    x: &[u64],
    mask: &Bitmask,
    start: usize,
    out: &mut [f64],
    out_mask: &mut Bitmask,
    scalar_body: &FScalar,
) where
    FScalar: Fn(u64) -> f64,
{
    let len = x.len();
    for idx in start..len {
        if !unsafe { mask.get_unchecked(idx) } {
            out[idx] = f64::NAN;
            unsafe { out_mask.set_unchecked(idx, false) };
        } else {
            let xi = unsafe { *x.get_unchecked(idx) };
            out[idx] = scalar_body(xi);
            unsafe { out_mask.set_unchecked(idx, true) };
        }
    }
}

/// Processes the scalar section for masked `u64->f64` kernels.
///
/// # Safety
/// Uses unchecked access for performance within validated ranges.
#[inline(always)]
pub fn process_scalar_masked_u64_std<FScalar>(
    x: &[u64],
    mask: &Bitmask,
    start: usize,
    out: &mut Vec64<f64>,
    out_mask: &mut Bitmask,
    scalar_body: &FScalar,
) where
    FScalar: Fn(u64) -> f64,
{
    let len = x.len();
    for idx in start..len {
        if !unsafe { mask.get_unchecked(idx) } {
            out.push(f64::NAN);
            unsafe { out_mask.set_unchecked(idx, false) };
        } else {
            let xi = unsafe { *x.get_unchecked(idx) };
            out.push(scalar_body(xi));
            unsafe { out_mask.set_unchecked(idx, true) };
        }
    }
}
