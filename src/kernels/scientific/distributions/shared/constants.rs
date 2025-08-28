// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Mathematical Constants Module** - *High-Precision Constants for Statistical Computing*
//!
//! Mathematical constants for high-performance statistical
//! distribution calculations with hard-coded precision. These constants support
//! accurate probability computations across the univariate distribution families.

// ******** Constants *************************************************************/
/// The square root of 2: √2 ≈ 1.414213562373095.
///
/// Fundamental mathematical constant used extensively in statistical computations,
/// particularly for normal distribution transformations and Box-Muller sampling.
/// Essential for error function calculations and variance scaling operations.
pub(crate) const SQRT_2: f64 = 1.4142135623730951_f64;

/// The square root of 2π: √(2π) ≈ 2.506628274631000.
///
/// Critical normalisation constant for probability density functions across
/// multiple statistical distribution families. Primary application in normal
/// distribution PDF normalisation and related multivariate computations.
pub(crate) const SQRT_2PI: f64 = 2.5066282746310002_f64;

/// The square root of π: √π ≈ 1.772453850905516.
///
/// Fundamental constant appearing in gamma function evaluations, error function
/// normalisations, and special function approximations across statistical computing.
/// Essential for half-integer gamma functions and related distribution parameters.
pub(crate) const SQRT_PI: f64 = 1.7724538509055159_f64;

/// Acklam's inverse normal CDF approximation coefficients (numerator polynomial).
///
/// High-precision rational function coefficients for computing the inverse standard
/// normal cumulative distribution function Φ⁻¹(p) using Peter John Acklam's
/// minimax rational approximation. Provides near-machine precision accuracy
/// across the central probability region 0.02425 < p < 0.97575.
pub(crate) const A: [f64; 6] = [
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285104469687e+02,
    1.383577518672690e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00,
];

/// Acklam's inverse normal CDF approximation coefficients (denominator polynomial).
///
/// Denominator coefficients for Peter John Acklam's rational function approximation
/// of the inverse standard normal cumulative distribution function. Used in conjunction
/// with the A array coefficients to form a complete minimax rational approximation
/// delivering near-machine precision accuracy for normal quantile computation.
pub(crate) const B: [f64; 5] = [
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01,
];

/// Acklam's inverse normal CDF approximation coefficients (tail region numerator).
///
/// Specialised rational function coefficients for computing inverse normal quantiles
/// in the extreme tail regions where p < 0.02425 or p > 0.97575. These coefficients
/// enable accurate quantile computation for probabilities corresponding to beyond
/// approximately ±2σ from the mean, critical for extreme value analysis.
pub(crate) const C: [f64; 6] = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00,
];
/// Acklam's inverse normal CDF approximation coefficients (tail region denominator).
///
/// Denominator polynomial coefficients for the extreme tail regions of Acklam's
/// inverse normal approximation. These coefficients complete the rational function
/// used when computing quantiles for probabilities p < 0.02425 or p > 0.97575,
/// ensuring high accuracy in the distribution's extreme regions.
pub(crate) const D: [f64; 4] = [
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00,
];

/// Lower probability threshold for Acklam's inverse normal CDF approximation.
///
/// Critical breakpoint probability separating the central rational approximation
/// from the specialised tail region approximation in Acklam's inverse normal algorithm.
/// Corresponds to approximately -2σ in the standard normal distribution, optimising
/// the balance between computational efficiency and numerical precision.
pub(crate) const P_LOW: f64 = 0.02425; // lower & upper break-points (≈ 2 σ) ; P_HIGH: f64 = 1.0 - P_LOW;

/// Lanczos approximation coefficients for high-precision gamma function evaluation.
///
/// Optimised coefficient array for the Lanczos approximation to the gamma function
/// with parameters g=7 (auxiliary parameter) and n=9 (number of terms). These
/// coefficients enable gamma function evaluation achieving near-machine precision
/// accuracy across the entire positive real domain and via reflection for negative arguments.
pub(crate) const COF: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_13,
    -176.615_029_162_140_59,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_571_6e-6,
    1.505_632_735_149_311_6e-7,
];

/// Natural logarithm of π: ln(π) ≈ 1.144729885849400.
///
/// Fundamental mathematical constant appearing in gamma function evaluations,
/// beta function computations, and various normalisation factors across
/// statistical distribution theory. Essential for reflection formula
/// implementations and multivariate distribution calculations.
pub const LN_PI: f64 = 1.1447298858494002; // Natural log of π

/// Reciprocal of π: 1/π ≈ 0.318309886183791.
///
/// Mathematical constant providing efficient division-by-π operations across
/// statistical computations. Enables multiplication-based alternatives to
/// division operations, improving computational performance and numerical
/// stability in probability density function evaluations.
pub const INV_PI: f64 = 1.0 / std::f64::consts::PI;

/// Half of the natural logarithm of 2π: ½ln(2π) ≈ 0.918938533204673.
///
/// Critical normalisation constant appearing throughout statistical distribution
/// theory, particularly in maximum likelihood estimation, information theory,
/// and multivariate probability computations. Essential component of the
/// Lanczos gamma function approximation and related special function evaluations.
pub const HALF_LOG_TWO_PI: f64 = 0.918_938_533_204_672_741_780_329_736_406;

/// High-precision factorial lookup table for n! computation.
///
/// Provides exact factorial values n! for n ∈ [0, 170] using pre-computed
/// double-precision floating-point representation. Enables O(1) factorial
/// computation eliminating expensive iterative multiplication or gamma function
/// evaluation for integer arguments within the representable range.
#[inline(always)]
pub fn factorial_lookup(n: u64) -> f64 {
    // Precomputed factorials for n=0..=170
    // Sourced via python's math lib float(math.factorial(n))
    const FACTORIALS: [f64; 171] = [
        1.0,
        1.0,
        2.0,
        6.0,
        24.0,
        120.0,
        720.0,
        5040.0,
        40320.0,
        362880.0,
        3628800.0,
        39916800.0,
        479001600.0,
        6227020800.0,
        87178291200.0,
        1307674368000.0,
        20922789888000.0,
        355687428096000.0,
        6402373705728000.0,
        1.21645100408832e17,
        2.43290200817664e18,
        5.109094217170944e19,
        1.1240007277776077e21,
        2.585201673888498e22,
        6.204484017332394e23,
        1.5511210043330986e25,
        4.0329146112660565e26,
        1.0888869450418352e28,
        3.0488834461171387e29,
        8.841761993739702e30,
        2.6525285981219107e32,
        8.222838654177922e33,
        2.631308369336935e35,
        8.683317618811886e36,
        2.9523279903960416e38,
        1.0333147966386145e40,
        3.7199332678990125e41,
        1.3763753091226346e43,
        5.230226174666011e44,
        2.0397882081197444e46,
        8.159152832478977e47,
        3.345252661316381e49,
        1.40500611775288e51,
        6.041526306337383e52,
        2.658271574788449e54,
        1.1962222086548019e56,
        5.502622159812089e57,
        2.5862324151116818e59,
        1.2413915592536073e61,
        6.082818640342675e62,
        3.0414093201713376e64,
        1.5511187532873822e66,
        8.065817517094388e67,
        4.2748832840600255e69,
        2.308436973392414e71,
        1.2696403353658276e73,
        7.109985878048635e74,
        4.0526919504877214e76,
        2.3505613312828785e78,
        1.3868311854568984e80,
        8.32098711274139e81,
        5.075802138772248e83,
        3.146997326038794e85,
        1.98260831540444e87,
        1.2688693218588417e89,
        8.247650592082472e90,
        5.443449390774431e92,
        3.647111091818868e94,
        2.4800355424368305e96,
        1.711224524281413e98,
        1.1978571669969892e100,
        8.504785885678623e101,
        6.1234458376886085e103,
        4.4701154615126844e105,
        3.307885441519386e107,
        2.48091408113954e109,
        1.8854947016660504e111,
        1.4518309202828587e113,
        1.1324281178206297e115,
        8.946182130782976e116,
        7.156945704626381e118,
        5.797126020747368e120,
        4.753643337012842e122,
        3.945523969720659e124,
        3.314240134565353e126,
        2.81710411438055e128,
        2.4227095383672734e130,
        2.107757298379528e132,
        1.8548264225739844e134,
        1.650795516090846e136,
        1.4857159644817615e138,
        1.352001527678403e140,
        1.2438414054641308e142,
        1.1567725070816416e144,
        1.087366156656743e146,
        1.032997848823906e148,
        9.916779348709496e149,
        9.619275968248212e151,
        9.426890448883248e153,
        9.332621544394415e155,
        9.332621544394415e157,
        9.42594775983836e159,
        9.614466715035127e161,
        9.90290071648618e163,
        1.0299016745145628e166,
        1.081396758240291e168,
        1.1462805637347084e170,
        1.226520203196138e172,
        1.324641819451829e174,
        1.4438595832024937e176,
        1.588245541522743e178,
        1.7629525510902446e180,
        1.974506857221074e182,
        2.2311927486598138e184,
        2.5435597334721877e186,
        2.925093693493016e188,
        3.393108684451898e190,
        3.969937160808721e192,
        4.684525849754291e194,
        5.574585761207606e196,
        6.689502913449127e198,
        8.094298525273444e200,
        9.875044200833601e202,
        1.214630436702533e205,
        1.506141741511141e207,
        1.882677176888926e209,
        2.372173242880047e211,
        3.0126600184576594e213,
        3.856204823625804e215,
        4.974504222477287e217,
        6.466855489220474e219,
        8.47158069087882e221,
        1.1182486511960043e224,
        1.4872707060906857e226,
        1.9929427461615188e228,
        2.6904727073180504e230,
        3.659042881952549e232,
        5.012888748274992e234,
        6.917786472619489e236,
        9.615723196941089e238,
        1.3462012475717526e241,
        1.898143759076171e243,
        2.695364137888163e245,
        3.854370717180073e247,
        5.5502938327393044e249,
        8.047926057471992e251,
        1.1749972043909107e254,
        1.727245890454639e256,
        2.5563239178728654e258,
        3.80892263763057e260,
        5.713383956445855e262,
        8.62720977423324e264,
        1.3113358856834524e267,
        2.0063439050956823e269,
        3.0897696138473508e271,
        4.789142901463394e273,
        7.471062926282894e275,
        1.1729568794264145e278,
        1.853271869493735e280,
        2.9467022724950384e282,
        4.7147236359920616e284,
        7.590705053947219e286,
        1.2296942187394494e289,
        2.0044015765453026e291,
        3.287218585534296e293,
        5.423910666131589e295,
        9.003691705778438e297,
        1.503616514864999e300,
        2.5260757449731984e302,
        4.269068009004705e304,
        7.257415615307999e306,
    ];
    FACTORIALS[n as usize]
}
