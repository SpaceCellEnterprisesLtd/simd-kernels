//! # **ToBit trait** - *IEEE 754 bit conversion*

// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

/// Generic trait for converting floating-point types to their IEEE 754 bit representation.
///
/// Unifies access to a `to_bits` method across different floating-point types,
/// enabling generic operation on floating-point bit patterns.
///
/// # Type Parameters
/// The associated `Bits` type represents the unsigned integer type with equivalent
/// bit width to the floating-point type, supporting equality comparison and hashing.
///
/// # Implementation Requirements
/// Implementations must preserve IEEE 754 bit layout and handle special values
/// (NaN, infinity, signed zero) according to standard specifications.
pub trait ToBits {
    /// The unsigned integer type representing the bit pattern.
    ///
    /// Must be the same bit width as the floating-point type and support
    /// equality comparison and hashing for use in collections.
    type Bits: Eq + std::hash::Hash + Copy;

    /// Converts the floating-point value to its IEEE 754 bit representation.
    ///
    /// Returns the raw bit pattern as an unsigned integer, preserving all
    /// floating-point special values and sign information.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use simd_kernels::traits::to_bits::ToBits;
    ///
    /// let f = 3.14f32;
    /// let bits = f.to_bits(); // Returns u32 bit representation
    /// ```
    fn to_bits(self) -> Self::Bits;
}

/// Implementation for 32-bit IEEE 754 single-precision floating-point values.
///
/// Maps to the corresponding 32-bit unsigned integer bit pattern, preserving
/// all floating-point special values including NaN bit patterns and signed zeros.
impl ToBits for f32 {
    type Bits = u32;

    #[inline(always)]
    fn to_bits(self) -> u32 {
        f32::to_bits(self)
    }
}

/// Implementation for 64-bit IEEE 754 double-precision floating-point values.
///
/// Maps to the corresponding 64-bit unsigned integer bit pattern, preserving
/// all floating-point special values including NaN bit patterns and signed zeros.
impl ToBits for f64 {
    type Bits = u64;

    #[inline(always)]
    fn to_bits(self) -> u64 {
        f64::to_bits(self)
    }
}
