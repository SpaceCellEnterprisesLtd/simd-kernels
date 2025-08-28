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
mod scipy_poisson_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::poisson::{
        poisson_cdf, poisson_pmf, poisson_quantile,
    };

    #[test]
    fn poisson_pmf_unit_rate() {
        let k = vec64![0, 1, 2, 3, 5, 10];
        let expect = vec64![
            0.36787944117144233,
            0.36787944117144233,
            0.18393972058572114,
            0.061313240195240391,
            0.00306566200976202,
            1.0137771196302987e-07
        ];
        let got = poisson_pmf(&k, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn poisson_pmf_small_rate() {
        let k = vec64![0, 1, 2, 3, 5, 10];
        let expect = vec64![
            0.1353352832366127,
            0.2706705664732254,
            0.2706705664732254,
            0.18044704431548356,
            0.036089408863096722,
            3.8189850648779541e-05
        ];
        let got = poisson_pmf(&k, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn poisson_pmf_medium_rate() {
        let k = vec64![0, 1, 2, 3, 5, 10];
        let expect = vec64![
            0.006737946999085467,
            0.033689734995427337,
            0.084224337488568321,
            0.1403738958142805,
            0.17546736976785068,
            0.018132788707821854
        ];
        let got = poisson_pmf(&k, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn poisson_pmf_large_rate() {
        let k = vec64![0, 1, 2, 3, 5, 10];
        let expect = vec64![
            4.5399929762484854e-05,
            0.00045399929762484861,
            0.0022699964881242435,
            0.007566654960414144,
            0.037833274802070792,
            0.12511003572113372
        ];
        let got = poisson_pmf(&k, 10.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn poisson_pmf_fractional_rate() {
        let k = vec64![0, 1, 2, 3, 5, 10];
        let expect = vec64![
            0.60653065971263342,
            0.30326532985631671,
            0.075816332464079192,
            0.012636055410679865,
            0.0001579506926334984,
            1.6322616219566214e-10
        ];
        let got = poisson_pmf(&k, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn poisson_pmf_large_lambda() {
        let k = vec64![45, 50, 55, 60, 65];
        let expect = vec64![
            0.045826241434197924,
            0.056325006325191662,
            0.042164352185115446,
            0.020104872145675377,
            0.006338637749006548
        ];
        let got = poisson_pmf(&k, 50.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn poisson_pmf_tiny_lambda() {
        let k = vec64![0, 1, 2];
        let expect = vec64![
            0.99004983374916811,
            0.009900498337491688,
            4.9502491687458457e-05
        ];
        let got = poisson_pmf(&k, 0.01, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    // We don't accept it

    // #[test]
    // fn poisson_pmf_negative_k() {
    //     let k = vec64![-2, -1, 0, 1, 2];
    //     let expect = vec64![0.0, 0.0, 0.1353352832366127, 0.2706705664732254, 0.2706705664732254];
    //     let got = poisson_pmf(&k, 2.0, None, None).unwrap();
    //     assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    // }

    #[test]
    fn poisson_pmf_zero_lambda() {
        let k = vec64![0, 1, 2, 10];
        let expect = vec64![1.0, 0.0, 0.0, 0.0];
        let got = poisson_pmf(&k, 0.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn poisson_cdf_cdf_unit() {
        let k = vec64![0, 1, 2, 3, 5, 10];
        let expect = vec64![
            0.36787944117144245,
            0.73575888234288467,
            0.91969860292860584,
            0.98101184312384615,
            0.99940581518241833,
            0.9999999899522336
        ];
        let got = poisson_cdf(&k, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn poisson_cdf_cdf_small() {
        let k = vec64![0, 1, 2, 3, 5, 10];
        let expect = vec64![
            0.1353352832366127,
            0.40600584970983794,
            0.6766764161830634,
            0.85712346049854704,
            0.98343639151938556,
            0.99999169177563152
        ];
        let got = poisson_cdf(&k, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn poisson_cdf_cdf_medium() {
        let k = vec64![0, 1, 2, 3, 5, 10];
        let expect = vec64![
            0.0067379469990854679,
            0.040427681994512792,
            0.12465201948308108,
            0.26502591529736158,
            0.61596065483306295,
            0.98630473140161712
        ];
        let got = poisson_cdf(&k, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn poisson_cdf_cdf_large() {
        let k = vec64![0, 1, 2, 3, 5, 10];
        let expect = vec64![
            4.5399929762484861e-05,
            0.00049939922738733355,
            0.0027693957155115775,
            0.010336050675925726,
            0.067085962879031888,
            0.58303975019298515
        ];
        let got = poisson_cdf(&k, 10.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn poisson_cdf_cdf_fractional() {
        let k = vec64![0, 1, 2, 3, 5, 10];
        let expect = vec64![
            0.60653065971263342,
            0.90979598956895014,
            0.98561232203302929,
            0.9982483774437092,
            0.99998583506267769,
            0.99999999999225919
        ];
        let got = poisson_cdf(&k, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn poisson_cdf_cdf_large_lambda() {
        let k = vec64![45, 50, 55, 60, 65];
        let expect = vec64![
            0.2668664740596442,
            0.53751669085314757,
            0.78447040069394969,
            0.92783982018674305,
            0.98273542507197664
        ];
        let got = poisson_cdf(&k, 50.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    // #[test]
    // fn poisson_cdf_cdf_zero_lambda() {
    //     let k = vec64![-3, -1, 0, 1, 5];
    //     let expect = vec64![0.0, 0.0, 1.0, 1.0, 1.0];
    //     let got = poisson_cdf(&k, 0.0, None, None).unwrap();
    //     assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    // }

    #[test]
    fn poisson_ppf_ppf_unit() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 4.0];
        let got = poisson_quantile(&q, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn poisson_ppf_ppf_small() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0];
        let got = poisson_quantile(&q, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn poisson_ppf_ppf_medium() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 11.0];
        let got = poisson_quantile(&q, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn poisson_ppf_ppf_large() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![3.0, 6.0, 8.0, 10.0, 12.0, 14.0, 18.0];
        let got = poisson_quantile(&q, 10.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn poisson_ppf_ppf_fractional() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 3.0];
        let got = poisson_quantile(&q, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn poisson_ppf_ppf_extreme() {
        let q = vec64![
            1e-10,
            1.0000000000000001e-05,
            0.5,
            0.99999000000000005,
            0.99999999989999999
        ];
        let expect = vec64![0.0, 0.0, 3.0, 13.0, 19.0];
        let got = poisson_quantile(&q, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-10);
    }

    #[test]
    fn poisson_ppf_ppf_zero_lambda() {
        let q = vec64![
            0.0,
            1.0000000000000001e-15,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            1.0
        ];
        let expect = vec64![-1.0, 0.0, 0.0, 0.0, 0.0, f64::INFINITY];
        let got = poisson_quantile(&q, 0.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }
}
