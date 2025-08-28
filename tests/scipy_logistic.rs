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
mod scipy_logistic_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::logistic::{
        logistic_cdf, logistic_pdf, logistic_quantile,
    };

    #[test]
    fn logistic_pdf_logistic_standard() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.04517665973091213,
            0.19661193324148185,
            0.25,
            0.19661193324148185,
            0.04517665973091213
        ];
        let got = logistic_pdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn logistic_pdf_logistic_shifted() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.006648056670790152,
            0.04517665973091213,
            0.10499358540350652,
            0.19661193324148185,
            0.19661193324148185
        ];
        let got = logistic_pdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn logistic_pdf_logistic_scaled() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.074573226035166418,
            0.11750185610079725,
            0.125,
            0.11750185610079725,
            0.074573226035166418
        ];
        let got = logistic_pdf(&x, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn logistic_pdf_logistic_negative() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.035325412426582221,
            0.5,
            0.20998717080701304,
            0.035325412426582221,
            0.00067047534151294815
        ];
        let got = logistic_pdf(&x, -1.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn logistic_pdf_logistic_large() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.034997861801168838,
            0.055030336550967916,
            0.065537311080493946,
            0.074719129967076206,
            0.083333333333333329
        ];
        let got = logistic_pdf(&x, 3.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn logistic_cdf_logistic_cdf_standard() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.047425873177566781,
            0.2689414213699951,
            0.5,
            0.7310585786300049,
            0.95257412682243336
        ];
        let got = logistic_cdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn logistic_cdf_logistic_cdf_shifted() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.0066928509242848554,
            0.047425873177566781,
            0.11920292202211755,
            0.2689414213699951,
            0.7310585786300049
        ];
        let got = logistic_cdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn logistic_cdf_logistic_cdf_scaled() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.18242552380635635,
            0.37754066879814541,
            0.5,
            0.62245933120185459,
            0.81757447619364365
        ];
        let got = logistic_cdf(&x, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn logistic_cdf_logistic_cdf_negative() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.017986209962091559,
            0.5,
            0.88079707797788231,
            0.98201379003790845,
            0.99966464986953363
        ];
        let got = logistic_cdf(&x, -1.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn logistic_cdf_logistic_cdf_large() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.11920292202211755,
            0.20860852732604496,
            0.2689414213699951,
            0.33924363123418283,
            0.5
        ];
        let got = logistic_cdf(&x, 3.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn logistic_ppf_logistic_ppf_standard() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            -4.5951198501345898,
            -2.1972245773362191,
            0.0,
            2.1972245773362196,
            4.5951198501345889
        ];
        let got = logistic_quantile(&q, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn logistic_ppf_logistic_ppf_shifted() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            -2.5951198501345898,
            -0.19722457733621912,
            2.0,
            4.19722457733622,
            6.5951198501345889
        ];
        let got = logistic_quantile(&q, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn logistic_ppf_logistic_ppf_scaled() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            -9.1902397002691796,
            -4.3944491546724382,
            0.0,
            4.3944491546724391,
            9.1902397002691778
        ];
        let got = logistic_quantile(&q, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn logistic_ppf_logistic_ppf_negative() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            -3.2975599250672949,
            -2.0986122886681096,
            -1.0,
            0.098612288668109782,
            1.2975599250672945
        ];
        let got = logistic_quantile(&q, -1.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn logistic_ppf_logistic_ppf_large() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            -10.785359550403768,
            -3.5916737320086574,
            3.0,
            9.5916737320086582,
            16.785359550403768
        ];
        let got = logistic_quantile(&q, 3.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }
}
