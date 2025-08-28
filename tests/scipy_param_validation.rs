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
mod scipy_param_validation_tests {
    use minarrow::vec64;

    // ---- Continuous distributions ----
    use simd_kernels::kernels::scientific::distributions::univariate::beta::beta_pdf;
    use simd_kernels::kernels::scientific::distributions::univariate::chi_squared::chi_square_pdf;
    use simd_kernels::kernels::scientific::distributions::univariate::exponential::exponential_pdf;
    use simd_kernels::kernels::scientific::distributions::univariate::gamma::gamma_pdf;
    use simd_kernels::kernels::scientific::distributions::univariate::normal::normal_pdf;

    #[test]
    fn beta_pdf_negative_alpha() {
        let x = vec64![0.5];
        let got = beta_pdf(&x, -1.0, 2.0, None, None);
        assert!(
            got.is_err(),
            "expected error for invalid parameters, got: {:?}",
            got
        );
    }

    #[test]
    fn beta_pdf_zero_beta() {
        let x = vec64![0.5];
        let got = beta_pdf(&x, 2.0, 0.0, None, None);
        assert!(
            got.is_err(),
            "expected error for invalid parameters, got: {:?}",
            got
        );
    }

    #[test]
    fn normal_pdf_negative_scale() {
        let x = vec64![0.5];
        let got = normal_pdf(&x, 0.0, -1.0, None, None);
        assert!(
            got.is_err(),
            "expected error for invalid parameters, got: {:?}",
            got
        );
    }

    #[test]
    fn normal_pdf_zero_scale() {
        let x = vec64![0.5];
        let got = normal_pdf(&x, 0.0, 0.0, None, None);
        assert!(
            got.is_err(),
            "expected error for invalid parameters, got: {:?}",
            got
        );
    }

    #[test]
    fn exponential_pdf_negative_scale() {
        let x = vec64![0.5];
        let got = exponential_pdf(&x, -2.0, None, None);
        assert!(
            got.is_err(),
            "expected error for invalid parameters, got: {:?}",
            got
        );
    }

    #[test]
    fn gamma_pdf_negative_shape() {
        let x = vec64![0.5];
        let got = gamma_pdf(&x, -1.0, 1.0, None, None);
        assert!(
            got.is_err(),
            "expected error for invalid parameters, got: {:?}",
            got
        );
    }

    #[test]
    fn gamma_pdf_negative_scale() {
        let x = vec64![0.5];
        let got = gamma_pdf(&x, 2.0, -1.0, None, None);
        assert!(
            got.is_err(),
            "expected error for invalid parameters, got: {:?}",
            got
        );
    }

    #[test]
    fn chi2_pdf_negative_df() {
        let x = vec64![0.5];
        let got = chi_square_pdf(&x, -1.0, None, None);
        assert!(
            got.is_err(),
            "expected error for invalid parameters, got: {:?}",
            got
        );
    }

    #[test]
    fn chi2_pdf_zero_df() {
        let x = vec64![0.5];
        let got = chi_square_pdf(&x, 0.0, None, None);
        assert!(
            got.is_err(),
            "expected error for invalid parameters, got: {:?}",
            got
        );
    }
}
