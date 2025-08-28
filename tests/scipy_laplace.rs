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
mod scipy_laplace_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::laplace::{
        laplace_cdf, laplace_pdf, laplace_quantile,
    };

    #[test]
    fn laplace_pdf_laplace_standard() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.024893534183931972,
            0.18393972058572117,
            0.5,
            0.18393972058572117,
            0.024893534183931972
        ];
        let got = laplace_pdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn laplace_pdf_laplace_shifted() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.0033689734995427335,
            0.024893534183931972,
            0.067667641618306351,
            0.18393972058572117,
            0.18393972058572117
        ];
        let got = laplace_pdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn laplace_pdf_laplace_scaled() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.055782540037107455,
            0.15163266492815836,
            0.25,
            0.15163266492815836,
            0.055782540037107455
        ];
        let got = laplace_pdf(&x, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn laplace_pdf_laplace_negative() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.018315638888734179,
            1.0,
            0.1353352832366127,
            0.018315638888734179,
            0.00033546262790251185
        ];
        let got = laplace_pdf(&x, -1.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn laplace_pdf_laplace_large() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.022555880539435452,
            0.043932856352621126,
            0.061313240195240391,
            0.085569519838765332,
            0.16666666666666666
        ];
        let got = laplace_pdf(&x, 3.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn laplace_cdf_laplace_cdf_standard() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.024893534183931972,
            0.18393972058572117,
            0.5,
            0.81606027941427883,
            0.97510646581606808
        ];
        let got = laplace_cdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn laplace_cdf_laplace_cdf_shifted() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.0033689734995427335,
            0.024893534183931972,
            0.067667641618306351,
            0.18393972058572117,
            0.81606027941427883
        ];
        let got = laplace_cdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn laplace_cdf_laplace_cdf_scaled() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.11156508007421491,
            0.30326532985631671,
            0.5,
            0.69673467014368329,
            0.88843491992578505
        ];
        let got = laplace_cdf(&x, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn laplace_cdf_laplace_cdf_negative() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.0091578194443670893,
            0.5,
            0.93233235838169359,
            0.99084218055563289,
            0.9998322686860488
        ];
        let got = laplace_cdf(&x, -1.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn laplace_cdf_laplace_cdf_large() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.067667641618306351,
            0.13179856905786339,
            0.18393972058572117,
            0.25670855951629601,
            0.5
        ];
        let got = laplace_cdf(&x, 3.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn laplace_ppf_laplace_ppf_standard() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            -3.912023005428146,
            -1.6094379124341003,
            0.0,
            1.6094379124341005,
            3.9120230054281451
        ];
        let got = laplace_quantile(&q, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn laplace_ppf_laplace_ppf_shifted() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            -1.912023005428146,
            0.39056208756589972,
            2.0,
            3.6094379124341005,
            5.9120230054281446
        ];
        let got = laplace_quantile(&q, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn laplace_ppf_laplace_ppf_scaled() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            -7.8240460108562919,
            -3.2188758248682006,
            0.0,
            3.218875824868201,
            7.8240460108562901
        ];
        let got = laplace_quantile(&q, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn laplace_ppf_laplace_ppf_negative() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            -2.9560115027140732,
            -1.8047189562170503,
            -1.0,
            -0.19528104378294975,
            0.95601150271407254
        ];
        let got = laplace_quantile(&q, -1.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn laplace_ppf_laplace_ppf_large() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            -8.7360690162844374,
            -1.8283137373023006,
            3.0,
            7.8283137373023015,
            14.736069016284436
        ];
        let got = laplace_quantile(&q, 3.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }
}
