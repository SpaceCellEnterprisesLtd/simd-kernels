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
mod scipy_cauchy_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::cauchy::{
        cauchy_cdf, cauchy_pdf, cauchy_quantile,
    };

    #[test]
    fn cauchy_pdf_standard() {
        let x = vec64![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.012242687930145796,
            0.063661977236758135,
            0.15915494309189535,
            0.31830988618379069,
            0.15915494309189535,
            0.063661977236758135,
            0.012242687930145796
        ];
        let got = cauchy_pdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn cauchy_pdf_shifted() {
        let x = vec64![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.0063661977236758142,
            0.018724110951987689,
            0.031830988618379068,
            0.063661977236758135,
            0.15915494309189535,
            0.31830988618379069,
            0.031830988618379068
        ];
        let got = cauchy_pdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn cauchy_pdf_scaled() {
        let x = vec64![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.021952405943709702,
            0.079577471545947673,
            0.12732395447351627,
            0.15915494309189535,
            0.12732395447351627,
            0.079577471545947673,
            0.021952405943709702
        ];
        let got = cauchy_pdf(&x, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn cauchy_pdf_negative_small_scale() {
        let x = vec64![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.037448221903975377,
            0.12732395447351627,
            0.037448221903975377,
            0.017205939793718417,
            0.009794150344116636,
            0.0063031660630453604,
            0.0024771197368388381
        ];
        let got = cauchy_pdf(&x, -3.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn cauchy_pdf_positive_large_scale() {
        let x = vec64![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.0087608225555171736,
            0.016464304457782273,
            0.021220659078919377,
            0.028086166427981531,
            0.038197186342054885,
            0.053051647697298449,
            0.1061032953945969
        ];
        let got = cauchy_pdf(&x, 5.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn cauchy_pdf_extreme_values() {
        let x = vec64![-100.0, -50.0, 0.0, 50.0, 100.0];
        let expect = vec64![
            3.1827805837795288e-05,
            0.00012727304525541412,
            0.31830988618379069,
            0.00012727304525541412,
            3.1827805837795288e-05
        ];
        let got = cauchy_pdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 9.9999999999999998e-13);
    }

    #[test]
    fn cauchy_pdf_tiny_scale() {
        let x = vec64![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.00012732344517973556,
            0.00079575482158893694,
            0.0031827805837795287,
            31.83098861837907,
            0.0031827805837795287,
            0.00079575482158893694,
            0.00012732344517973556
        ];
        let got = cauchy_pdf(&x, 0.0, 0.01, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn cauchy_cdf_cdf_standard() {
        let x = vec64![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.062832958189001184,
            0.14758361765043326,
            0.25,
            0.5,
            0.75,
            0.85241638234956674,
            0.93716704181099897
        ];
        let got = cauchy_cdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn cauchy_cdf_cdf_shifted() {
        let x = vec64![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.045167235300866554,
            0.077979130377369324,
            0.10241638234956672,
            0.14758361765043326,
            0.25,
            0.5,
            0.89758361765043326
        ];
        let got = cauchy_cdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn cauchy_cdf_cdf_scaled() {
        let x = vec64![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.12111894159084341,
            0.25,
            0.35241638234956674,
            0.5,
            0.64758361765043326,
            0.75,
            0.87888105840915665
        ];
        let got = cauchy_cdf(&x, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn cauchy_cdf_cdf_negative() {
        let x = vec64![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.077979130377369324,
            0.85241638234956674,
            0.92202086962263063,
            0.94743154328874657,
            0.96041657583943452,
            0.96827448256944648,
            0.98013147569445924
        ];
        let got = cauchy_cdf(&x, -3.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn cauchy_cdf_cdf_positive_large() {
        let x = vec64![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.092773579077742349,
            0.12888105840915659,
            0.14758361765043326,
            0.17202086962263069,
            0.20483276469913345,
            0.25,
            0.5
        ];
        let got = cauchy_cdf(&x, 5.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn cauchy_cdf_cdf_extreme() {
        let x = vec64![-100.0, -50.0, 0.0, 50.0, 100.0];
        let expect = vec64![
            0.0031829927649082552,
            0.0063653491009727963,
            0.5,
            0.99363465089902725,
            0.99681700723509181
        ];
        let got = cauchy_cdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 9.9999999999999998e-13);
    }

    #[test]
    fn cauchy_cdf_cdf_tiny_scale() {
        let x = vec64![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.00063661892354325536,
            0.0015915361682059693,
            0.0031829927649082552,
            0.5,
            0.99681700723509181,
            0.99840846383179405,
            0.99936338107645684
        ];
        let got = cauchy_cdf(&x, 0.0, 0.01, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn cauchy_ppf_ppf_standard() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            -31.820515953773956,
            -3.077683537175254,
            -1.0000000000000002,
            0.0,
            1.0000000000000002,
            3.0776835371752544,
            31.820515953773928
        ];
        let got = cauchy_quantile(&q, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn cauchy_ppf_ppf_shifted() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            -29.820515953773956,
            -1.077683537175254,
            0.99999999999999978,
            2.0,
            3.0,
            5.077683537175254,
            33.820515953773928
        ];
        let got = cauchy_quantile(&q, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn cauchy_ppf_ppf_scaled() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            -63.641031907547912,
            -6.155367074350508,
            -2.0000000000000004,
            0.0,
            2.0000000000000004,
            6.1553670743505089,
            63.641031907547855
        ];
        let got = cauchy_quantile(&q, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn cauchy_ppf_ppf_negative() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            -18.910257976886978,
            -4.538841768587627,
            -3.5,
            -3.0,
            -2.5,
            -1.4611582314123728,
            12.910257976886964
        ];
        let got = cauchy_quantile(&q, -3.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn cauchy_ppf_ppf_extreme() {
        let q = vec64![
            1e-10,
            1.0000000000000001e-05,
            0.5,
            0.99999000000000005,
            0.99999999989999999
        ];
        let expect = vec64![
            -3183098861.8379064,
            -31830.988607907086,
            0.0,
            31830.988608051954,
            3183098598.4671478
        ];
        let got = cauchy_quantile(&q, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-08);
    }

    #[test]
    fn cauchy_ppf_ppf_tiny_scale() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            -0.31820515953773959,
            -0.030776835371752541,
            -0.010000000000000002,
            0.0,
            0.010000000000000002,
            0.030776835371752544,
            0.31820515953773926
        ];
        let got = cauchy_quantile(&q, 0.0, 0.01, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }
}
