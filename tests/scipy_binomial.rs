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
mod scipy_binomial_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::binomial::{
        binomial_cdf, binomial_pmf, binomial_quantile,
    };
    // use simd_kernels::kernels::scientific::distributions::discrete::binomial::{binomial_pmf, binomial_cdf, binomial_quantile};

    #[test]
    fn binomial_pmf_fair() {
        let k = vec64![0, 2, 5, 8, 10];
        let expect = vec64![
            0.00097656249999999892,
            0.043945312500000042,
            0.24609375000000003,
            0.043945312500000042,
            0.0009765625
        ];
        let got = binomial_pmf(&k, 10, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn binomial_pmf_skewed_right() {
        let k = vec64![0, 2, 5, 8, 10];
        let expect = vec64![
            0.028247524900000001,
            0.2334744405000001,
            0.10291934519999989,
            0.0014467004999999982,
            5.9048999999999975e-06
        ];
        let got = binomial_pmf(&k, 10, 0.29999999999999999, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn binomial_pmf_skewed_left() {
        let k = vec64![0, 2, 5, 8, 10];
        let expect = vec64![
            5.9049000000000093e-06,
            0.0014467005000000004,
            0.10291934519999998,
            0.23347444050000013,
            0.028247524899999984
        ];
        let got = binomial_pmf(&k, 10, 0.69999999999999996, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn binomial_pmf_larger_n() {
        let k = vec64![0, 2, 5, 8, 10];
        let expect = vec64![
            9.5367431640625064e-07,
            0.00018119812011718736,
            0.01478576660156255,
            0.1201343536376954,
            0.17619705200195296
        ];
        let got = binomial_pmf(&k, 20, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn binomial_pmf_small_n_p() {
        let k = vec64![0, 2, 5, 8, 10];
        let expect = vec64![
            0.3276799999999998,
            0.20479999999999987,
            0.00032000000000000008,
            0.0,
            0.0
        ];
        let got = binomial_pmf(&k, 5, 0.20000000000000001, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn binomial_pmf_large_n() {
        let k = vec64![40, 45, 50, 55, 60];
        let expect = vec64![
            0.010843866711637978,
            0.048474296626430713,
            0.079589237387178755,
            0.04847429662643072,
            0.010843866711637959
        ];
        let got = binomial_pmf(&k, 100, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn binomial_pmf_small_p() {
        let k = vec64![0, 1, 2, 3];
        let expect = vec64![
            0.90438207500880408,
            0.091351724748364005,
            0.0041523511249256405,
            0.00011184784174883875
        ];
        let got = binomial_pmf(&k, 10, 0.01, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn binomial_pmf_large_p() {
        let k = vec64![7, 8, 9, 10];
        let expect = vec64![
            0.00011184784174883898,
            0.0041523511249256517,
            0.091351724748364116,
            0.90438207500880441
        ];
        let got = binomial_pmf(&k, 10, 0.98999999999999999, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    // We don't accept negative k

    // #[test]
    // fn binomial_pmf_out_of_range() {
    //     let k = vec64![-1, 0, 5, 10, 11];
    //     let expect = vec64![0.0, 0.00097656249999999892, 0.24609375000000003, 0.0009765625, 0.0];
    //     let got = binomial_pmf(&k, 10, 0.5, None, None).unwrap();
    //     assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    // }

    #[test]
    fn binomial_pmf_bernoulli() {
        let k = vec64![0, 1];
        let expect = vec64![0.70000000000000018, 0.29999999999999999];
        let got = binomial_pmf(&k, 1, 0.29999999999999999, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn binomial_cdf_cdf_fair() {
        let k = vec64![0, 1, 2, 3, 4, 5];
        let expect = vec64![
            0.0009765625,
            0.0107421875,
            0.0546875,
            0.171875,
            0.376953125,
            0.623046875
        ];
        let got = binomial_cdf(&k, 10, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn binomial_cdf_cdf_skewed_right() {
        let k = vec64![0, 1, 2, 3, 4, 5];
        let expect = vec64![
            0.028247524900000005,
            0.14930834589999992,
            0.3827827863999998,
            0.64961071840000018,
            0.84973166740000006,
            0.95265101260000007
        ];
        let got = binomial_cdf(&k, 10, 0.29999999999999999, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn binomial_cdf_cdf_larger_n() {
        let k = vec64![0, 1, 2, 3, 4, 5];
        let expect = vec64![
            9.5367431640625e-07,
            2.002716064453125e-05,
            0.00020122528076171875,
            0.0012884140014648438,
            0.005908966064453125,
            0.020694732666015625
        ];
        let got = binomial_cdf(&k, 20, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn binomial_cdf_cdf_small_n_large_p() {
        let k = vec64![0, 1, 2, 3, 4, 5];
        let expect = vec64![
            0.00031999999999999965,
            0.0067199999999999942,
            0.057919999999999965,
            0.2627199999999999,
            0.67232000000000003,
            1.0
        ];
        let got = binomial_cdf(&k, 5, 0.80000000000000004, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn binomial_cdf_cdf_medium() {
        let k = vec64![0, 1, 2, 3, 4, 5];
        let expect = vec64![
            0.00047018498457599962,
            0.0051720348303359977,
            0.027114000777215989,
            0.090501902401535966,
            0.21727770565017596,
            0.40321555041484797
        ];
        let got = binomial_cdf(&k, 15, 0.40000000000000002, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn binomial_cdf_cdf_large_n() {
        let k = vec64![40, 45, 50, 55, 60];
        let expect = vec64![
            0.028443966820490392,
            0.18410080866334788,
            0.53979461869358947,
            0.86437348796308233,
            0.98239989989114762
        ];
        let got = binomial_cdf(&k, 100, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-13);
    }

    // We don't accept negative k

    // #[test]
    // fn binomial_cdf_cdf_negative_k() {
    //     let k = vec64![-2, -1, 0, 1, 2];
    //     let expect = vec64![0.0, 0.0, 0.0009765625, 0.0107421875, 0.0546875];
    //     let got = binomial_cdf(&k, 10, 0.5, None, None).unwrap();
    //     assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    // }

    #[test]
    fn binomial_cdf_cdf_beyond_n() {
        let k = vec64![8, 9, 10, 11, 12];
        let expect = vec64![0.9892578125, 0.9990234375, 1.0, 1.0, 1.0];
        let got = binomial_cdf(&k, 10, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn binomial_ppf_ppf_fair() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0];
        let got = binomial_quantile(&q, 10, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn binomial_ppf_ppf_skewed() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0];
        let got = binomial_quantile(&q, 10, 0.29999999999999999, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn binomial_ppf_ppf_larger_n() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![5.0, 7.0, 8.0, 10.0, 12.0, 13.0, 15.0];
        let got = binomial_quantile(&q, 20, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn binomial_ppf_ppf_large_skewed() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 17.0];
        let got = binomial_quantile(&q, 50, 0.20000000000000001, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn binomial_ppf_ppf_small_n() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 5.0];
        let got = binomial_quantile(&q, 5, 0.80000000000000004, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn binomial_ppf_ppf_extreme() {
        let q = vec64![
            1e-10,
            1.0000000000000001e-05,
            0.5,
            0.99999000000000005,
            0.99999999989999999
        ];
        let expect = vec64![0.0, 0.0, 5.0, 10.0, 10.0];
        let got = binomial_quantile(&q, 10, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-10);
    }

    #[test]
    fn binomial_ppf_ppf_boundaries() {
        let q = vec64![0.0, 1e-300, 0.5, 1.0, 1.0];
        let expect = vec64![-1.0, 0.0, 6.0, 15.0, 15.0];
        let got = binomial_quantile(&q, 15, 0.40000000000000002, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }
}
