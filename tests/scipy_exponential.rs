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
mod scipy_exponential_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::exponential::{
        exponential_cdf, exponential_pdf, exponential_quantile,
    };

    #[test]
    fn exponential_pdf_rate_0_5() {
        let x = vec64![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.5,
            0.38940039153570244,
            0.30326532985631671,
            0.18393972058572117,
            0.0410424993119494,
            0.0033689734995427335
        ];
        let got = exponential_pdf(&x, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_pdf_rate_1_0() {
        let x = vec64![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            1.0,
            0.60653065971263342,
            0.36787944117144233,
            0.1353352832366127,
            0.006737946999085467,
            4.5399929762484854e-05
        ];
        let got = exponential_pdf(&x, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_pdf_rate_2_0() {
        let x = vec64![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            2.0,
            0.73575888234288467,
            0.2706705664732254,
            0.036631277777468357,
            9.0799859524969708e-05,
            4.1223072448771157e-09
        ];
        let got = exponential_pdf(&x, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_pdf_rate_5_0() {
        let x = vec64![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            5.0,
            0.41042499311949399,
            0.03368973499542733,
            0.00022699964881242425,
            6.9439719324820095e-11,
            9.6437492398195884e-22
        ];
        let got = exponential_pdf(&x, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_pdf_rate_10_0() {
        let x = vec64![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            10.0,
            0.06737946999085466,
            0.0004539992976248485,
            2.0611536224385576e-08,
            1.9287498479639177e-21,
            3.720075976020836e-43
        ];
        let got = exponential_pdf(&x, 10.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_pdf_small_rate() {
        let x = vec64![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.01,
            0.0099501247919268239,
            0.0099004983374916811,
            0.0098019867330675532,
            0.0095122942450071406,
            0.009048374180359595
        ];
        let got = exponential_pdf(&x, 0.01, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_pdf_large_rate() {
        let x = vec64![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            100.0,
            1.9287498479639178e-20,
            3.7200759760208363e-42,
            1.3838965267367375e-85,
            7.1245764067412853e-216,
            0.0
        ];
        let got = exponential_pdf(&x, 100.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_pdf_negative_values() {
        let x = vec64![-10.0, -1.0, -0.10000000000000001, 0.0, 1.0];
        let expect = vec64![0.0, 0.0, 0.0, 1.0, 0.36787944117144233];
        let got = exponential_pdf(&x, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn exponential_pdf_extreme_values() {
        let x = vec64![0.0, 10.0, 50.0, 100.0, 500.0];
        let expect = vec64![
            1.0,
            4.5399929762484854e-05,
            1.9287498479639178e-22,
            3.7200759760208361e-44,
            7.1245764067412855e-218
        ];
        let got = exponential_pdf(&x, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-12);
    }

    #[test]
    fn exponential_cdf_cdf_rate_0_5() {
        let x = vec64![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.0,
            0.22119921692859515,
            0.39346934028736658,
            0.63212055882855767,
            0.91791500137610116,
            0.99326205300091452
        ];
        let got = exponential_cdf(&x, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_cdf_cdf_rate_1_0() {
        let x = vec64![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.0,
            0.39346934028736658,
            0.63212055882855767,
            0.8646647167633873,
            0.99326205300091452,
            0.99995460007023751
        ];
        let got = exponential_cdf(&x, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_cdf_cdf_rate_2_0() {
        let x = vec64![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.0,
            0.63212055882855767,
            0.8646647167633873,
            0.98168436111126578,
            0.99995460007023751,
            0.99999999793884642
        ];
        let got = exponential_cdf(&x, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_cdf_cdf_rate_5_0() {
        let x = vec64![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.0,
            0.91791500137610116,
            0.99326205300091452,
            0.99995460007023751,
            0.99999999998611211,
            1.0
        ];
        let got = exponential_cdf(&x, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_cdf_cdf_rate_10_0() {
        let x = vec64![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.0,
            0.99326205300091452,
            0.99995460007023751,
            0.99999999793884642,
            1.0,
            1.0
        ];
        let got = exponential_cdf(&x, 10.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_cdf_cdf_small_rate() {
        let x = vec64![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.0,
            0.0049875208073176863,
            0.0099501662508319454,
            0.019801326693244699,
            0.048770575499285998,
            0.095162581964040441
        ];
        let got = exponential_cdf(&x, 0.01, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_cdf_cdf_negative() {
        let x = vec64![-10.0, -1.0, -0.10000000000000001, 0.0, 1.0];
        let expect = vec64![0.0, 0.0, 0.0, 0.0, 0.63212055882855767];
        let got = exponential_cdf(&x, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn exponential_cdf_cdf_approaching_one() {
        let x = vec64![5.0, 10.0, 20.0, 50.0, 100.0];
        let expect = vec64![
            0.99326205300091452,
            0.99995460007023751,
            0.99999999793884642,
            1.0,
            1.0
        ];
        let got = exponential_cdf(&x, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-12);
    }

    #[test]
    fn exponential_ppf_ppf_rate_0_5() {
        let q = vec64![
            0.001,
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999,
            0.999
        ];
        let expect = vec64![
            0.002001000667167067,
            0.020100671707002884,
            0.21072103131565262,
            0.5753641449035618,
            1.3862943611198906,
            2.7725887222397811,
            4.6051701859880918,
            9.2103403719761818,
            13.815510557964272
        ];
        let got = exponential_quantile(&q, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_ppf_ppf_rate_1_0() {
        let q = vec64![
            0.001,
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999,
            0.999
        ];
        let expect = vec64![
            0.0010005003335835335,
            0.010050335853501442,
            0.10536051565782631,
            0.2876820724517809,
            0.69314718055994529,
            1.3862943611198906,
            2.3025850929940459,
            4.6051701859880909,
            6.9077552789821359
        ];
        let got = exponential_quantile(&q, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_ppf_ppf_rate_2_0() {
        let q = vec64![
            0.001,
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999,
            0.999
        ];
        let expect = vec64![
            0.00050025016679176675,
            0.005025167926750721,
            0.052680257828913155,
            0.14384103622589045,
            0.34657359027997264,
            0.69314718055994529,
            1.151292546497023,
            2.3025850929940455,
            3.453877639491068
        ];
        let got = exponential_quantile(&q, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_ppf_ppf_rate_5_0() {
        let q = vec64![
            0.001,
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999,
            0.999
        ];
        let expect = vec64![
            0.00020010006671670672,
            0.0020100671707002885,
            0.021072103131565264,
            0.05753641449035618,
            0.13862943611198905,
            0.2772588722239781,
            0.46051701859880922,
            0.92103403719761823,
            1.3815510557964272
        ];
        let got = exponential_quantile(&q, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_ppf_ppf_rate_10_0() {
        let q = vec64![
            0.001,
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999,
            0.999
        ];
        let expect = vec64![
            0.00010005003335835336,
            0.0010050335853501442,
            0.010536051565782632,
            0.02876820724517809,
            0.069314718055994526,
            0.13862943611198905,
            0.23025850929940461,
            0.46051701859880911,
            0.69077552789821361
        ];
        let got = exponential_quantile(&q, 10.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_ppf_ppf_extreme() {
        let q = vec64![
            1e-10,
            1.0000000000000001e-05,
            0.5,
            0.99999000000000005,
            0.99999999989999999
        ];
        let expect = vec64![
            1.00000000005e-10,
            1.0000050000333337e-05,
            0.69314718055994529,
            11.51292546497478,
            23.02585084720009
        ];
        let got = exponential_quantile(&q, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-10);
    }

    #[test]
    fn exponential_ppf_ppf_small_rate() {
        let q = vec64![
            0.001,
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999,
            0.999
        ];
        let expect = vec64![
            0.10005003335835334,
            1.0050335853501442,
            10.53605156578263,
            28.76820724517809,
            69.314718055994533,
            138.62943611198907,
            230.25850929940458,
            460.5170185988091,
            690.77552789821357
        ];
        let got = exponential_quantile(&q, 0.01, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn exponential_ppf_ppf_boundaries() {
        let q = vec64![0.0, 1e-300, 0.5, 1.0, 1.0];
        let expect = vec64![
            0.0,
            5.0000000000000001e-301,
            0.34657359027997264,
            f64::INFINITY,
            f64::INFINITY
        ];
        let got = exponential_quantile(&q, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }
}
