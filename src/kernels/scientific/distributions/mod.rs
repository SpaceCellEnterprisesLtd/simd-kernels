// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Statistical Distributions Module** - *Comprehensive Probability Distribution Computing*
//!
//! Advanced statistical distribution kernels providing high-performance probability density
//! functions (PDFs), cumulative distribution functions (CDFs), quantile functions, and
//! random sampling with SIMD acceleration and numerical precision guarantees.
//!
//! ## Distribution Categories
//! - **Univariate distributions**: Complete coverage of common continuous and discrete distributions
//! - **Multivariate distributions**: Multivariate normal, Student-t, Wishart, and advanced distributions
//! - **Parametric families**: Beta, gamma, normal, exponential, and related distribution families  
//! - **Discrete distributions**: Binomial, Poisson, geometric, and hypergeometric variants
//!
//! ## Core Statistical Functions  
//! Each distribution provides a complete statistical interface:
//! - **Probability density/mass functions**: Optimised PDF/PMF evaluation with numerical stability
//! - **Cumulative distribution functions**: CDF computation with extended precision algorithms
//! - **Quantile functions**: Inverse CDF calculation using robust bracketing and refinement
//! - **Random sampling**: High-quality pseudorandom generation with distributional correctness
//!
//! ## Computational Architecture
//! Distribution calculations employ sophisticated numerical techniques for accuracy and performance:
//! - **SIMD vectorisation**: Hardware-accelerated evaluation of distribution functions
//! - **Rational approximations**: Optimised polynomial and rational function approximations
//! - **Series expansions**: Convergent series with adaptive truncation for transcendental functions
//! - **Numerical integration**: Gauss-Kronrod quadrature for complex distribution functions
//!
//! ## Arrow Integration and Null Handling
//! The module integrates seamlessly with Apache Arrow's memory model and null semantics:
//! - **Null-aware processing**: Efficient handling of missing values with validity bitmasks
//! - **SIMD-accelerated masking**: Vectorised null propagation without conditional branches
//! - **Arrow-compatible layouts**: Direct operation on Arrow array structures
//! - **Memory efficiency**: Zero-copy operations where mathematically valid
//!
//! ### Null Value Philosophy
//! Rather than assume, we choose to recognise inf and NaN as valid float values
//! (consistent with Apache Arrow semantics), leaving it to the user to subsequently
//! treat them as nulls if they wish, given that there are numerical scenarios where
//! they represent information gain. This approach avoids computational overhead in
//! the hot path whilst preserving mathematical correctness for edge cases.
//!
//! ## Numerical Precision and Stability
//! All distribution implementations prioritise numerical accuracy across parameter ranges.
//! See `./tests` for any specific tolerance requirements, where it is measured against Scipy.
//! Whilst these pass on the development machine, platform specific difference may impact your
//! test results, and thus one should keep this in mind when evaluating this library's fit for your use case.
//!
//! ## Disclaimer
//! This implementation is provided on a best-effort basis and is intended for
//! general scientific and engineering use. While every attempt has been made to
//! match the accuracy and behaviour of established libraries such as SciPy, we
//! make no guarantees as to correctness, fitness for any particular purpose, or
//! suitability for uses such as in life-critical, safety-critical, or financial applications.
//!
//! Results may differ from other libraries due to platform, compiler, or implementation
//! differences. Edge cases and special values are handled explicitly for compatibility
//! with SciPy (v1.16) but users are responsible for independently verifying that this
//! function meets their accuracy and reliability requirements.
//!
//! By using these functions, you accept all responsibility for outcomes or decisions
//! based upon its results.

/// # **Shared Distribution Utilities** - *Common Infrastructure for Distribution Computing*
///
/// Foundational utilities, constants, and helper functions shared across all probability
/// distributions, providing consistent numerical methods and sampling infrastructure.
///
/// This module contains the core mathematical building blocks that enable efficient
/// and accurate distribution computation across all statistical functions.
///
/// ## Modules
/// - **`constants`**: Mathematical constants and precomputed values
/// - **`sampler`**: Random number generation and sampling utilities  
/// - **`scalar`**: Special functions and mathematical utilities
pub mod shared {
    pub mod constants;
    pub mod sampler;
    pub mod scalar;
}

/// # **Univariate Distributions** - *Single-Variable Probability Distributions*
///
/// Complete collection of univariate probability distributions covering both continuous
/// and discrete families with comprehensive statistical function implementations.
///
/// Each distribution provides PDF/PMF, CDF, quantile functions, and random sampling
/// with SIMD acceleration and numerical precision guarantees.
///
/// ## Distribution Categories
/// - **Continuous**: beta, cauchy, chi-squared, exponential, gamma, laplace, logistic, lognormal, normal, student_t, uniform, weibull
/// - **Discrete**: binomial, discrete_uniform, geometric, hypergeometric, multinomial, neg_binomial, poisson
/// - **Common utilities**: Shared patterns and mathematical building blocks
pub mod univariate {
    // common kernel patterns
    pub mod common;

    // distributions
    pub mod beta;
    pub mod binomial;
    pub mod cauchy;
    pub mod chi_squared;
    /// Discrete uniform distribution kernels - equal probability over finite integer range.
    pub mod discrete_uniform;
    /// Exponential distribution kernels - continuous distribution for inter-arrival times.
    pub mod exponential;
    pub mod gamma;
    pub mod geometric;
    pub mod gumbell;
    pub mod hypergeometric;
    pub mod laplace;
    pub mod logistic;
    pub mod lognormal;
    pub mod multinomial;
    pub mod neg_binomial;
    pub mod normal;
    pub mod poisson;
    pub mod student_t;
    pub mod uniform;
    pub mod weibull;
}

#[cfg(feature = "linear_algebra")]
pub mod multivariate;
