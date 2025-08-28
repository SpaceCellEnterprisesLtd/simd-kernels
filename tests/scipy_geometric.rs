// This file is created by gen_scipy_tests.py and contains reference
// tests whose expected values are produced by SciPy at generation time.
//

// Each test compares our kernel outputs against SciPy with a per-test tolerance.
// NaN/Inf equality is handled by util::assert_slice_close.

mod util;
#[cfg(feature = "probability_distributions")]
mod scipy_geometric_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::geometric::{
        geometric_cdf, geometric_pmf, geometric_quantile,
    };

    // TODO[5]: Fix source Including Non-Decimal 'k'
    // Re-running the Scipy file generation will ruin this file

    // use simd_kernels::kernels::scientific::distributions::discrete::geometric::{geometric_pmf, geometric_cdf, geometric_quantile};

    #[test]
    fn geometric_pmf_high_prob() {
        let k = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20];
        let expect = vec64![
            0.80000000000000004,
            0.15999999999999998,
            0.031999999999999987,
            0.006399999999999996,
            0.001279999999999999,
            0.00025599999999999972,
            5.1199999999999937e-05,
            1.0239999999999985e-05,
            2.0479999999999963e-06,
            4.0959999999999922e-07,
            1.310719999999996e-10,
            4.1943039999999827e-14
        ];
        let got = geometric_pmf(&k, 0.8, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_pmf_medium_prob() {
        let k = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20];
        let expect = vec64![
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.0078125,
            0.00390625,
            0.001953125,
            0.0009765625,
            3.0517578125e-05,
            9.5367431640625e-07
        ];
        let got = geometric_pmf(&k, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_pmf_low_prob() {
        let k = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20];
        let expect = vec64![
            0.10000000000000001,
            0.090000000000000011,
            0.081000000000000016,
            0.072900000000000006,
            0.065610000000000002,
            0.059049000000000011,
            0.053144100000000007,
            0.047829690000000008,
            0.04304672100000001,
            0.038742048900000013,
            0.02287679245496101,
            0.013508517176729929
        ];
        let got = geometric_pmf(&k, 0.1, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_pmf_very_high_prob() {
        let k = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20];
        let expect = vec64![
            0.98999999999999999,
            0.0099000000000000095,
            9.9000000000000184e-05,
            9.9000000000000259e-07,
            9.9000000000000341e-09,
            9.9000000000000434e-11,
            9.9000000000000507e-13,
            9.9000000000000607e-15,
            9.9000000000000709e-17,
            9.9000000000000783e-19,
            9.9000000000001225e-29,
            9.9000000000001672e-39
        ];
        let got = geometric_pmf(&k, 0.99, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_pmf_very_low_prob() {
        let k = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20];
        let expect = vec64![
            0.01,
            0.0099000000000000008,
            0.0098010000000000007,
            0.0097029899999999999,
            0.0096059600999999998,
            0.0095099004989999993,
            0.00941480149401,
            0.0093206534790699,
            0.0092274469442792002,
            0.0091351724748364085,
            0.0086874581276897827,
            0.008261686238355867
        ];
        let got = geometric_pmf(&k, 0.01, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_pmf_degenerate() {
        let k = vec64![1, 2, 3];
        let expect = vec64![1.0, 0.0, 0.0];
        let got = geometric_pmf(&k, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_pmf_extended_range() {
        let k = vec64![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        ];
        let expect = vec64![
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.0078125,
            0.00390625,
            0.001953125,
            0.0009765625,
            0.00048828125,
            0.000244140625,
            0.0001220703125,
            6.103515625e-05,
            3.0517578125e-05,
            1.52587890625e-05,
            7.62939453125e-06,
            3.814697265625e-06,
            1.9073486328125e-06,
            9.5367431640625e-07,
            4.76837158203125e-07
        ];
        let got = geometric_pmf(&k, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_pmf_rare_event() {
        let k = vec64![1, 101, 501, 1001, 2001];
        let expect = vec64![
            0.001,
            0.00090479214711370898,
            0.00060637894486118471,
            0.00036769542477096376,
            0.00013519992539749945
        ];
        let got = geometric_pmf(&k, 0.001, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn geometric_pmf_quality_control() {
        let k = vec64![1, 6, 11, 16, 21, 26, 31, 41, 51];
        let expect = vec64![
            0.050000000000000003,
            0.038689046874999994,
            0.029936846961918936,
            0.023164561507987652,
            0.017924296120427095,
            0.013869478656091689,
            0.010731938197146865,
            0.0064256078282551561,
            0.0038472487638356576
        ];
        let got = geometric_pmf(&k, 0.05, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_cdf_high_prob_cdf() {
        let x = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20];
        let expect = vec64![
            0.80000000000000004,
            0.95999999999999996,
            0.99199999999999999,
            0.99839999999999995,
            0.99968000000000001,
            0.99993600000000005,
            0.99998719999999996,
            0.99999744000000002,
            0.99999948800000005,
            0.99999989759999997,
            0.99999999996723199,
            0.99999999999998956
        ];
        let got = geometric_cdf(&x, 0.8, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_cdf_medium_prob_cdf() {
        let x = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20];
        let expect = vec64![
            0.5,
            0.75,
            0.875,
            0.9375,
            0.96875,
            0.984375,
            0.9921875,
            0.99609375,
            0.998046875,
            0.9990234375,
            0.999969482421875,
            0.99999904632568359
        ];
        let got = geometric_cdf(&x, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_cdf_low_prob_cdf() {
        let x = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20];
        let expect = vec64![
            0.10000000000000001,
            0.19,
            0.27100000000000002,
            0.34389999999999998,
            0.40951000000000004,
            0.468559,
            0.52170309999999998,
            0.56953279000000001,
            0.61257951100000008,
            0.65132155990000007,
            0.79410886790535096,
            0.87842334540943079
        ];
        let got = geometric_cdf(&x, 0.1, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_cdf_very_high_cdf() {
        let x = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20];
        let expect = vec64![
            0.98999999999999999,
            0.99990000000000001,
            0.99999899999999997,
            0.99999998999999995,
            0.99999999989999999,
            0.99999999999900002,
            0.99999999999999001,
            0.99999999999999989,
            1.0,
            1.0,
            1.0,
            1.0
        ];
        let got = geometric_cdf(&x, 0.99, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_cdf_degenerate_cdf() {
        let x = vec64![0, 1, 2];
        let expect = vec64![0.0, 1.0, 1.0];
        let got = geometric_cdf(&x, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_cdf_extended_cdf() {
        let x = vec64![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
        ];
        let expect = vec64![
            0.5,
            0.75,
            0.875,
            0.9375,
            0.96875,
            0.984375,
            0.9921875,
            0.99609375,
            0.998046875,
            0.9990234375,
            0.99951171875,
            0.999755859375,
            0.9998779296875,
            0.99993896484375,
            0.999969482421875,
            0.9999847412109375,
            0.99999237060546875,
            0.99999618530273438,
            0.99999809265136719,
            0.99999904632568359,
            0.9999995231628418
        ];
        let got = geometric_cdf(&x, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_cdf_rare_cdf() {
        let x = vec64![1, 101, 501, 1001, 2001];
        let expect = vec64![
            0.001,
            0.096112645033404664,
            0.39422743408367616,
            0.63267227065380693,
            0.86493527452789787
        ];
        let got = geometric_cdf(&x, 0.001, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn geometric_cdf_qc_cdf() {
        let x = vec64![1, 6, 11, 16, 21, 26, 31, 41, 51];
        let expect = vec64![
            0.050000000000000003,
            0.26490810937500003,
            0.43119990772354,
            0.55987333134823436,
            0.65943837371188496,
            0.73647990553425768,
            0.79609317425420933,
            0.87791345126315179,
            0.9269022734871224
        ];
        let got = geometric_cdf(&x, 0.05, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_ppf_high_prob_ppf() {
        let q = vec64![
            0.01,
            0.050000000000000003,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.94999999999999996,
            0.98999999999999999
        ];
        let expect = vec64![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let got = geometric_quantile(&q, 0.8, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_ppf_medium_prob_ppf() {
        let q = vec64![
            0.01,
            0.050000000000000003,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.94999999999999996,
            0.98999999999999999
        ];
        let expect = vec64![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 5.0, 7.0];
        let got = geometric_quantile(&q, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_ppf_low_prob_ppf() {
        let q = vec64![
            0.01,
            0.050000000000000003,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.94999999999999996,
            0.98999999999999999
        ];
        let expect = vec64![1.0, 1.0, 1.0, 3.0, 7.0, 14.0, 22.0, 29.0, 44.0];
        let got = geometric_quantile(&q, 0.1, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_ppf_very_high_ppf() {
        let q = vec64![
            0.01,
            0.050000000000000003,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.94999999999999996,
            0.98999999999999999
        ];
        let expect = vec64![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let got = geometric_quantile(&q, 0.99, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_ppf_very_low_ppf() {
        let q = vec64![
            0.01,
            0.050000000000000003,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.94999999999999996,
            0.98999999999999999
        ];
        let expect = vec64![1.0, 6.0, 11.0, 29.0, 69.0, 138.0, 230.0, 299.0, 459.0];
        let got = geometric_quantile(&q, 0.01, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_ppf_edge_quantiles() {
        let q = vec64![0.0, 0.001, 0.999, 1.0];
        // SciPy geom.ppf(0, p) = 1
        let expect = vec64![1.0, 1.0, 20.0, f64::INFINITY];
        let got = geometric_quantile(&q, 0.3, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0e-15);
    }

    #[test]
    fn geometric_ppf_coin_flip_ppf() {
        let q = vec64![
            0.01,
            0.050000000000000003,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.94999999999999996,
            0.98999999999999999
        ];
        let expect = vec64![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 5.0, 7.0];
        let got = geometric_quantile(&q, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn geometric_ppf_qc_ppf() {
        let q = vec64![
            0.01,
            0.050000000000000003,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.94999999999999996,
            0.98999999999999999
        ];
        let expect = vec64![1.0, 1.0, 3.0, 6.0, 14.0, 28.0, 45.0, 59.0, 90.0];
        let got = geometric_quantile(&q, 0.05, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }
}
