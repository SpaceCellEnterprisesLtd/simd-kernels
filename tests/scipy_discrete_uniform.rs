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
mod scipy_discrete_uniform_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::discrete_uniform::{
        discrete_uniform_cdf, discrete_uniform_pmf, discrete_uniform_quantile,
    };
    // use simd_kernels::kernels::scientific::distributions::discrete::discrete_uniform::{discrete_uniform_pmf, discrete_uniform_cdf, discrete_uniform_quantile};

    #[test]
    fn discrete_uniform_pmf_die_roll() {
        let k = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expect = vec64![
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.0,
            0.0,
            0.0,
            0.0
        ];
        let got = discrete_uniform_pmf(&k, 1, 7, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_pmf_coin_flip() {
        let k = vec64![0, 1, 2];
        let expect = vec64![0.5, 0.5, 0.0];
        let got = discrete_uniform_pmf(&k, 0, 2, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_pmf_large_range() {
        let k = vec64![1, 25, 50, 75, 100, 101];
        let expect = vec64![0.01, 0.01, 0.01, 0.01, 0.01, 0.0];
        let got = discrete_uniform_pmf(&k, 1, 101, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_pmf_single_value() {
        let k = vec64![5, 6, 7];
        let expect = vec64![0.0, 1.0, 0.0];
        let got = discrete_uniform_pmf(&k, 6, 7, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    // Doesn't take negative values
    // #[test]
    // fn discrete_uniform_pmf_negative_range() {
    //     let k = vec64![-5, -3, -1, 0, 1, 3];
    //     let expect = vec64![0.0, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.0];
    //     let got = discrete_uniform_pmf(&k, -4, 2, None, None).unwrap();
    //     assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    // }

    #[test]
    fn discrete_uniform_pmf_out_of_bounds() {
        let k = vec64![0, 1, 2, 3, 4, 8, 9, 10];
        let expect = vec64![
            0.0,
            0.0,
            0.20000000000000001,
            0.20000000000000001,
            0.20000000000000001,
            0.0,
            0.0,
            0.0
        ];
        let got = discrete_uniform_pmf(&k, 2, 7, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_pmf_edge_boundaries() {
        let k = vec64![10, 11, 20, 21];
        let expect = vec64![
            0.090909090909090912,
            0.090909090909090912,
            0.090909090909090912,
            0.0
        ];
        let got = discrete_uniform_pmf(&k, 10, 21, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_pmf_huge_range() {
        let k = vec64![1, 500, 1000, 1001, 1500];
        let expect = vec64![0.001, 0.001, 0.001, 0.0, 0.0];
        let got = discrete_uniform_pmf(&k, 1, 1001, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_cdf_die_cdf() {
        let x = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expect = vec64![
            0.16666666666666666,
            0.33333333333333331,
            0.5,
            0.66666666666666663,
            0.83333333333333337,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        ];
        let got = discrete_uniform_cdf(&x, 1, 7, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_cdf_coin_cdf() {
        let x = vec64![0, 1, 2];
        let expect = vec64![0.5, 1.0, 1.0];
        let got = discrete_uniform_cdf(&x, 0, 2, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_cdf_large_cdf() {
        let x = vec64![1, 25, 50, 75, 100, 101];
        let expect = vec64![0.01, 0.25, 0.5, 0.75, 1.0, 1.0];
        let got = discrete_uniform_cdf(&x, 1, 101, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_cdf_negative_cdf() {
        let x = vec64![-5, -3, -1, 0, 1, 3];
        let expect = vec64![
            0.0,
            0.33333333333333331,
            0.66666666666666663,
            0.83333333333333337,
            1.0,
            1.0
        ];
        let got = discrete_uniform_cdf(&x, -4, 2, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_cdf_oob_cdf() {
        let x = vec64![0, 1, 2, 3, 4, 8, 9, 10];
        let expect = vec64![
            0.0,
            0.0,
            0.20000000000000001,
            0.40000000000000002,
            0.59999999999999998,
            1.0,
            1.0,
            1.0
        ];
        let got = discrete_uniform_cdf(&x, 2, 7, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_cdf_edge_cdf() {
        let x = vec64![10, 11, 20, 21];
        let expect = vec64![0.090909090909090912, 0.18181818181818182, 1.0, 1.0];
        let got = discrete_uniform_cdf(&x, 10, 21, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_cdf_single_cdf() {
        let x = vec64![5, 6, 7];
        let expect = vec64![0.0, 1.0, 1.0];
        let got = discrete_uniform_cdf(&x, 6, 7, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_cdf_cumulative() {
        let x = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expect = vec64![
            0.20000000000000001,
            0.40000000000000002,
            0.59999999999999998,
            0.80000000000000004,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        ];
        let got = discrete_uniform_cdf(&x, 1, 6, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    // does not accept fractional

    // #[test]
    // fn discrete_uniform_cdf_fractional() {
    //     let x = vec64![1.2, 2.7000000000000002, 3.8999999999999999, 4.0999999999999996, 5.5];
    //     let expect = vec64![0.16666666666666666, 0.33333333333333331, 0.5, 0.66666666666666663, 0.83333333333333337];
    //     let got = discrete_uniform_cdf(&x, 1, 7, None, None).unwrap();
    //     assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    // }

    #[test]
    fn discrete_uniform_ppf_die_ppf() {
        let q = vec64![
            0.0,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999,
            1.0
        ];
        let expect = vec64![0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 6.0, 6.0];
        let got = discrete_uniform_quantile(&q, 1, 7, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_ppf_coin_ppf() {
        let q = vec64![
            0.0,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999,
            1.0
        ];
        let expect = vec64![-1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let got = discrete_uniform_quantile(&q, 0, 2, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_ppf_large_ppf() {
        let q = vec64![
            0.0,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999,
            1.0
        ];
        let expect = vec64![0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 99.0, 100.0];
        let got = discrete_uniform_quantile(&q, 1, 101, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_ppf_negative_ppf() {
        let q = vec64![
            0.0,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999,
            1.0
        ];
        let expect = vec64![-5.0, -4.0, -3.0, -2.0, 0.0, 1.0, 1.0, 1.0];
        let got = discrete_uniform_quantile(&q, -4, 2, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_ppf_edge_ppf() {
        let q = vec64![
            0.0,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999,
            1.0
        ];
        let expect = vec64![9.0, 11.0, 12.0, 15.0, 18.0, 19.0, 20.0, 20.0];
        let got = discrete_uniform_quantile(&q, 10, 21, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_ppf_decimal_ppf() {
        let q = vec64![
            0.0,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999,
            1.0
        ];
        let expect = vec64![0.0, 1.0, 3.0, 5.0, 8.0, 9.0, 10.0, 10.0];
        let got = discrete_uniform_quantile(&q, 1, 11, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_ppf_single_ppf() {
        let q = vec64![
            0.0,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999,
            1.0
        ];
        let expect = vec64![5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0];
        let got = discrete_uniform_quantile(&q, 6, 7, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_ppf_huge_ppf() {
        let q = vec64![
            0.0,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999,
            1.0
        ];
        let expect = vec64![0.0, 100.0, 250.0, 500.0, 750.0, 900.0, 990.0, 1000.0];
        let got = discrete_uniform_quantile(&q, 1, 1001, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn discrete_uniform_pmf_duniform_standard() {
        let k = vec64![1, 3, 5, 7];
        let expect = vec64![
            0.1111111111111111,
            0.1111111111111111,
            0.1111111111111111,
            0.1111111111111111
        ];
        let got = discrete_uniform_pmf(&k, 1, 10, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn discrete_uniform_pmf_duniform_zero_start() {
        let k = vec64![1, 3];
        let expect = vec64![0.20000000000000001, 0.20000000000000001];
        let got = discrete_uniform_pmf(&k, 0, 5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn discrete_uniform_pmf_duniform_symmetric() {
        let k = vec64![1, 3];
        let expect = vec64![0.10000000000000001, 0.10000000000000001];
        let got = discrete_uniform_pmf(&k, -5, 5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn discrete_uniform_pmf_duniform_large() {
        let k = vec64![10];
        let expect = vec64![0.10000000000000001];
        let got = discrete_uniform_pmf(&k, 10, 20, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn discrete_uniform_cdf_duniform_cdf_standard() {
        let x = vec64![1, 3, 5, 7, 10];
        let expect = vec64![
            0.1111111111111111,
            0.33333333333333331,
            0.55555555555555558,
            0.77777777777777779,
            1.0
        ];
        let got = discrete_uniform_cdf(&x, 1, 10, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn discrete_uniform_cdf_duniform_cdf_zero_start() {
        let x = vec64![1, 3, 5];
        let expect = vec64![0.40000000000000002, 0.80000000000000004, 1.0];
        let got = discrete_uniform_cdf(&x, 0, 5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn discrete_uniform_cdf_duniform_cdf_symmetric() {
        let x = vec64![1, 3, 5];
        let expect = vec64![0.69999999999999996, 0.90000000000000002, 1.0];
        let got = discrete_uniform_cdf(&x, -5, 5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn discrete_uniform_ppf_duniform_ppf_standard() {
        let q = vec64![0.0, 0.10000000000000001, 0.5, 0.90000000000000002, 1.0];
        let expect = vec64![0.0, 1.0, 5.0, 9.0, 9.0];
        let got = discrete_uniform_quantile(&q, 1, 10, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn discrete_uniform_ppf_duniform_ppf_zero_start() {
        let q = vec64![0.0, 0.10000000000000001, 0.5, 0.90000000000000002, 1.0];
        let expect = vec64![-1.0, 0.0, 2.0, 4.0, 4.0];
        let got = discrete_uniform_quantile(&q, 0, 5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn discrete_uniform_ppf_duniform_ppf_symmetric() {
        let q = vec64![0.0, 0.10000000000000001, 0.5, 0.90000000000000002, 1.0];
        let expect = vec64![-6.0, -5.0, -1.0, 3.0, 4.0];
        let got = discrete_uniform_quantile(&q, -5, 5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }
}
