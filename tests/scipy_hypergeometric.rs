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
mod scipy_hypergeometric_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::hypergeometric::{
        hypergeometric_cdf, hypergeometric_pmf, hypergeometric_quantile,
    };
    //use simd_kernels::kernels::scientific::distributions::discrete::hypergeometric::{hypergeometric_pmf, hypergeometric_cdf, hypergeometric_quantile};

    #[test]
    fn hypergeometric_pmf_urn_problem() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expect = vec64![
            0.00010319917440660477,
            0.0043343653250774005,
            0.047678018575851404,
            0.19865841073271417,
            0.35758513931888553,
            0.28606811145510841,
            0.095356037151702794,
            0.010216718266253873,
            0.0,
            0.0,
            0.0
        ];
        let got = hypergeometric_pmf(&k, 20, 7, 12, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn hypergeometric_pmf_quality_control() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expect = vec64![
            0.58375236692615196,
            0.33939091100357671,
            0.07021880917315379,
            0.0063835281066503451,
            0.00025103762217164283,
            3.3471682956219036e-06,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ];
        let got = hypergeometric_pmf(&k, 100, 5, 10, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-13);
    }

    #[test]
    fn hypergeometric_pmf_small_population() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expect = vec64![
            0.083333333333333329,
            0.41666666666666669,
            0.41666666666666669,
            0.083333333333333329,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ];
        let got = hypergeometric_pmf(&k, 10, 3, 5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn hypergeometric_pmf_large_population() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expect = vec64![
            0.0044757909778860884,
            0.02629724428840241,
            0.074863809884413157,
            0.13761600691531059,
            0.18366258065775778,
            0.18972022366892588,
            0.15791548991134316,
            0.10887455680502003,
            0.063430849746980641,
            0.031703117762557638,
            0.01375399213629099
        ];
        let got = hypergeometric_pmf(&k, 1000, 100, 50, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-12);
    }

    #[test]
    fn hypergeometric_pmf_draw_all() {
        let k = vec64![0, 1, 2, 3];
        let expect = vec64![0.0, 0.0, 0.0, 1.0];
        let got = hypergeometric_pmf(&k, 5, 3, 5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn hypergeometric_pmf_single_draw() {
        let k = vec64![0, 1, 2];
        let expect = vec64![0.59999999999999998, 0.40000000000000002, 0.0];
        let got = hypergeometric_pmf(&k, 10, 4, 1, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn hypergeometric_pmf_equal_success_draw() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expect = vec64![
            5.4125441122345148e-06,
            0.00054125441122345147,
            0.010960401827274893,
            0.077940635216177015,
            0.23869319534954206,
            0.34371820130334058,
            0.23869319534954211,
            0.077940635216177015,
            0.010960401827274893,
            0.00054125441122345147,
            5.4125441122345148e-06
        ];
        let got = hypergeometric_pmf(&k, 20, 10, 10, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn hypergeometric_pmf_high_success_rate() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expect = vec64![
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.12307692307692306,
            0.43076923076923074,
            0.36923076923076925,
            0.076923076923076927,
            0.0,
            0.0
        ];
        let got = hypergeometric_pmf(&k, 15, 12, 8, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn hypergeometric_cdf_urn_cdf() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expect = vec64![
            0.00010319917440660477,
            0.0044375644994840051,
            0.052115583075335412,
            0.2507739938080496,
            0.6083591331269349,
            0.89442724458204337,
            0.98978328173374608,
            1.0,
            1.0,
            1.0,
            1.0
        ];
        let got = hypergeometric_cdf(&k, 20, 7, 12, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn hypergeometric_cdf_qc_cdf() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expect = vec64![
            0.58375236692615184,
            0.92314327792972861,
            0.99336208710288243,
            0.99974561520953276,
            0.99999665283170436,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        ];
        let got = hypergeometric_cdf(&k, 100, 5, 10, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-13);
    }

    #[test]
    fn hypergeometric_cdf_small_cdf() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expect = vec64![
            0.083333333333333329,
            0.5,
            0.91666666666666663,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        ];
        let got = hypergeometric_cdf(&k, 10, 3, 5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn hypergeometric_cdf_large_cdf() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expect = vec64![
            0.0044757909778860884,
            0.030773035266288502,
            0.10563684515070164,
            0.24325285206601233,
            0.42691543272376992,
            0.61663565639269602,
            0.77455114630403921,
            0.88342570310905921,
            0.94685655285603987,
            0.9785596706185975,
            0.99231366275488853
        ];
        let got = hypergeometric_cdf(&k, 1000, 100, 50, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-12);
    }

    #[test]
    fn hypergeometric_cdf_draw_all_cdf() {
        let k = vec64![0, 1, 2, 3];
        let expect = vec64![0.0, 0.0, 0.0, 1.0];
        let got = hypergeometric_cdf(&k, 5, 3, 5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn hypergeometric_cdf_single_cdf() {
        let k = vec64![0, 1, 2];
        let expect = vec64![0.59999999999999998, 1.0, 1.0];
        let got = hypergeometric_cdf(&k, 10, 4, 1, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn hypergeometric_cdf_equal_cdf() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expect = vec64![
            5.4125441122345148e-06,
            0.00054666695533568601,
            0.011507068782610578,
            0.089447703998787598,
            0.32814089934832957,
            0.67185910065167032,
            0.91055229600121246,
            0.98849293121738946,
            0.99945333304466433,
            0.99999458745588776,
            1.0
        ];
        let got = hypergeometric_cdf(&k, 20, 10, 10, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn hypergeometric_cdf_high_rate_cdf() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expect = vec64![
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.12307692307692306,
            0.55384615384615388,
            0.92307692307692313,
            1.0,
            1.0,
            1.0
        ];
        let got = hypergeometric_cdf(&k, 15, 12, 8, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn hypergeometric_ppf_urn_ppf() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let got = hypergeometric_quantile(&q, 20, 7, 12, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn hypergeometric_ppf_qc_ppf() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0];
        let got = hypergeometric_quantile(&q, 100, 5, 10, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn hypergeometric_ppf_small_ppf() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let got = hypergeometric_quantile(&q, 10, 3, 5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn hypergeometric_ppf_large_ppf() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0];
        let got = hypergeometric_quantile(&q, 1000, 100, 50, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn hypergeometric_ppf_draw_all_ppf() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0];
        let got = hypergeometric_quantile(&q, 5, 3, 5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn hypergeometric_ppf_single_ppf() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let got = hypergeometric_quantile(&q, 10, 4, 1, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn hypergeometric_ppf_equal_ppf() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![2.0, 4.0, 4.0, 5.0, 6.0, 6.0, 8.0];
        let got = hypergeometric_quantile(&q, 20, 10, 10, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn hypergeometric_ppf_high_rate_ppf() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0];
        let got = hypergeometric_quantile(&q, 15, 12, 8, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn hypergeometric_pmf_hypergeometric_pmf_standard() {
        let k = vec64![0, 1, 2, 3, 4, 5];
        let expect = vec64![
            0.00010319917440660477,
            0.0043343653250774005,
            0.047678018575851404,
            0.19865841073271417,
            0.35758513931888553,
            0.28606811145510841
        ];
        let got = hypergeometric_pmf(&k, 20, 7, 12, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn hypergeometric_pmf_hypergeometric_pmf_large() {
        let k = vec64![0, 1, 2, 3, 4, 5];
        let expect = vec64![
            0.0029248638425452612,
            0.027855846119478677,
            0.10825794741888303,
            0.22592962939592984,
            0.28005860310537134,
            0.21508500718492518
        ];
        let got = hypergeometric_pmf(&k, 50, 10, 20, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn hypergeometric_pmf_hypergeometric_pmf_half() {
        let k = vec64![0, 1, 2, 3, 4, 5];
        let expect = vec64![
            0.0039682539682539689,
            0.099206349206349229,
            0.39682539682539691,
            0.39682539682539691,
            0.099206349206349229,
            0.0039682539682539689
        ];
        let got = hypergeometric_pmf(&k, 10, 5, 5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn hypergeometric_cdf_hypergeometric_cdf_standard() {
        let k = vec64![0, 1, 2, 3, 4, 5];
        let expect = vec64![
            0.00010319917440660477,
            0.0044375644994840051,
            0.052115583075335412,
            0.2507739938080496,
            0.6083591331269349,
            0.89442724458204337
        ];
        let got = hypergeometric_cdf(&k, 20, 7, 12, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn hypergeometric_cdf_hypergeometric_cdf_large() {
        let k = vec64![0, 1, 2, 3, 4, 5];
        let expect = vec64![
            0.0029248638425452612,
            0.03078070996202394,
            0.13903865738090695,
            0.36496828677683679,
            0.64502688988220802,
            0.86011189706713331
        ];
        let got = hypergeometric_cdf(&k, 50, 10, 20, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn hypergeometric_cdf_hypergeometric_cdf_half() {
        let k = vec64![0, 1, 2, 3, 4, 5];
        let expect = vec64![
            0.0039682539682539689,
            0.1031746031746032,
            0.50000000000000011,
            0.89682539682539675,
            0.99603174603174605,
            1.0
        ];
        let got = hypergeometric_cdf(&k, 10, 5, 5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn hypergeometric_ppf_hypergeometric_ppf_standard() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![2.0, 3.0, 4.0, 6.0, 7.0];
        let got = hypergeometric_quantile(&q, 20, 7, 12, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn hypergeometric_ppf_hypergeometric_ppf_large() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![1.0, 2.0, 4.0, 6.0, 7.0];
        let got = hypergeometric_quantile(&q, 50, 10, 20, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn hypergeometric_ppf_hypergeometric_ppf_half() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![1.0, 1.0, 2.0, 4.0, 4.0];
        let got = hypergeometric_quantile(&q, 10, 5, 5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }
}
