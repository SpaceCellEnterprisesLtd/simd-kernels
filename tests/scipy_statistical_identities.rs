// AUTO-GENERATED FILE. DO NOT EDIT BY HAND.
// Generated from SciPy 1.16.1 on 2025-08-21T22:25:23Z.
// Distribution: Beta
//
// This file is created by gen_scipy_tests.py and contains reference
// tests whose expected values are produced by SciPy at generation time.
//

// Each test compares our kernel outputs against SciPy with a per-test tolerance.
// NaN/Inf equality is handled by util::assert_slice_close.

mod util;
#[cfg(feature = "probability_distributions")]
mod scipy_statistical_identities_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;

    // Kernels we rely on here
    use simd_kernels::kernels::scientific::distributions::univariate::chi_squared::chi_square_pdf;
    use simd_kernels::kernels::scientific::distributions::univariate::exponential::exponential_pdf;
    use simd_kernels::kernels::scientific::distributions::univariate::gamma::gamma_pdf;
    use simd_kernels::kernels::scientific::distributions::univariate::geometric::geometric_pmf;
    use simd_kernels::kernels::scientific::distributions::univariate::neg_binomial::neg_binomial_pmf;
    use simd_kernels::kernels::scientific::distributions::univariate::weibull::weibull_pdf;

    #[test]
    fn chi2_gamma_identity_chi2() {
        let x = vec64![0.5, 1.0, 2.0, 4.0, 8.0];
        let expect = vec64![
            0.09735009788392562,
            0.15163266492815836,
            0.18393972058572114,
            0.1353352832366127,
            0.036631277777468364
        ];
        let got = chi_square_pdf(&x, 4.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn chi2_gamma_identity_gamma() {
        let x = vec64![0.5, 1.0, 2.0, 4.0, 8.0];
        let expect = vec64![
            0.09735009788392561,
            0.15163266492815836,
            0.18393972058572117,
            0.1353352832366127,
            0.036631277777468364
        ];
        let got = gamma_pdf(&x, 2.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn weibull_expon_identity_weibull() {
        let x = vec64![0.5, 1.0, 2.0, 4.0, 8.0];
        let expect = vec64![
            0.6065306597126334,
            0.36787944117144233,
            0.1353352832366127,
            0.01831563888873418,
            0.00033546262790251185
        ];
        let got = weibull_pdf(&x, 1.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn weibull_expon_identity_expon() {
        let x = vec64![0.5, 1.0, 2.0, 4.0, 8.0];
        let expect = vec64![
            0.6065306597126334,
            0.36787944117144233,
            0.1353352832366127,
            0.01831563888873418,
            0.00033546262790251185
        ];
        let got = exponential_pdf(&x, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn nbinom_geom_identity_nbinom() {
        let k = vec64![1, 2, 3, 5, 10, 15];
        let expect = vec64![
            0.21000000000000005,
            0.14700000000000002,
            0.10289999999999998,
            0.050421,
            0.00847425747,
            0.0014242684529828917
        ];
        let got = neg_binomial_pmf(&k, 1, 0.3, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn nbinom_geom_identity_geom() {
        let k = vec64![2, 3, 4, 6, 11, 16];
        let expect = vec64![
            0.21,
            0.14699999999999996,
            0.10289999999999998,
            0.05042099999999998,
            0.008474257469999994,
            0.0014242684529828986
        ];
        let got = geometric_pmf(&k, 0.3, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }
}
