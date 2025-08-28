// Generated from SciPy 1.16.1 on 2025-08-21T22:25:23Z.
// Distribution: Beta
//
// This file is created by gen_scipy_tests.py and contains reference
// tests whose expected values are produced by SciPy at generation time.

// Each test compares our kernel outputs against SciPy with a per-test tolerance.
// NaN/Inf equality is handled by util::assert_slice_close.

mod util;
#[cfg(feature = "probability_distributions")]
mod scipy_chi2_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::chi_squared::{
        chi_square_cdf, chi_square_pdf, chi_square_quantile,
    };

    // TODO[5]: Fix source - re-running the Scipy file generation will ruin this file
    // use simd_kernels::kernels::scientific::distributions::univariate::chi2::{chi2_pdf, chi2_cdf, chi_square_quantile};
    // + .to_values() + df float

    #[test]
    fn chi2_pdf_df_1() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            1.2000389484301359,
            0.43939128946772243,
            0.24197072451914337,
            0.10377687435514868,
            0.014644982561926487,
            0.00085003666025203336
        ];
        let got = chi_square_pdf(&x, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn chi2_pdf_df_2() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.47561471225035701,
            0.38940039153570244,
            0.30326532985631671,
            0.18393972058572114,
            0.041042499311949393,
            0.0033689734995427331
        ];
        let got = chi_square_pdf(&x, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn chi2_pdf_df_3() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.12000389484301362,
            0.21969564473386122,
            0.24197072451914337,
            0.20755374871029739,
            0.073224912809632448,
            0.0085003666025203432
        ];
        let got = chi_square_pdf(&x, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn chi2_pdf_df_5() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.0040001298281004552,
            0.03661594078897687,
            0.080656908173047798,
            0.1383691658068649,
            0.12204152134938738,
            0.028334555341734475
        ];
        let got = chi_square_pdf(&x, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn chi2_pdf_df_10() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            1.2385799798186382e-07,
            6.3378969976514042e-05,
            0.00078975346316749137,
            0.0076641550244050498,
            0.0668009428905426,
            0.087733684883925411
        ];
        let got = chi_square_pdf(&x, 10.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn chi2_pdf_large_df() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            1.3889308509414044e-127,
            2.0200026568141581e-93,
            8.8562141121619759e-79,
            3.0239224849774924e-64,
            2.1290671364112124e-45,
            9.8383651913941531e-32
        ];
        let got = chi_square_pdf(&x, 100.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 9.9999999999999998e-13);
    }

    #[test]
    fn chi2_pdf_near_zero() {
        let x = vec64![
            0.0,
            1e-10,
            1.0000000000000001e-05,
            0.01,
            0.10000000000000001
        ];
        let expect = vec64![
            0.5,
            0.499999999975,
            0.49999750000624998,
            0.49750623959634116,
            0.47561471225035701
        ];
        let got = chi_square_pdf(&x, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-10);
    }

    #[test]
    fn chi2_pdf_extreme_values() {
        let x = vec64![0.01, 1.0, 10.0, 50.0, 100.0];
        let expect = vec64![
            0.00013231751582567066,
            0.080656908173047798,
            0.028334555341734475,
            6.5295277212572104e-10,
            2.564866208902142e-20
        ];
        let got = chi_square_pdf(&x, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn chi2_pdf_negative_values() {
        let x = vec64![-10.0, -1.0, -0.10000000000000001, 0.0, 1.0];
        let expect = vec64![0.0, 0.0, 0.0, 0.0, 0.24197072451914337];
        let got = chi_square_pdf(&x, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn chi2_cdf_cdf_df_1() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.24817036595415076,
            0.52049987781304663,
            0.68268949213708585,
            0.84270079294971512,
            0.97465268132253169,
            0.9984345977419975
        ];
        let got = chi_square_cdf(&x, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn chi2_cdf_cdf_df_2() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.048770575499285991,
            0.22119921692859512,
            0.39346934028736652,
            0.63212055882855767,
            0.91791500137610116,
            0.99326205300091452
        ];
        let got = chi_square_cdf(&x, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn chi2_cdf_cdf_df_3() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.0081625762681235194,
            0.081108588345324181,
            0.19874804309879915,
            0.42759329552912023,
            0.82820285570326646,
            0.9814338645369568
        ];
        let got = chi_square_cdf(&x, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn chi2_cdf_cdf_df_5() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.0001623166119226152,
            0.007876706767370404,
            0.037434226752703609,
            0.15085496391539038,
            0.58411981300449201,
            0.92476475385348778
        ];
        let got = chi_square_cdf(&x, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn chi2_cdf_cdf_df_10() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            2.4979513360065075e-09,
            6.6117105610342441e-06,
            0.00017211562995584072,
            0.0036598468273437131,
            0.10882198108584877,
            0.55950671493478787
        ];
        let got = chi_square_cdf(&x, 10.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn chi2_cdf_cdf_large_df() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            2.7805877168288313e-130,
            2.0299524618646476e-95,
            1.7887765104351563e-80,
            1.2337508979097585e-65,
            2.2386989282289475e-46,
            2.1810592140785249e-32
        ];
        let got = chi_square_cdf(&x, 100.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 9.9999999999999998e-13);
    }

    #[test]
    fn chi2_cdf_cdf_near_zero() {
        let x = vec64![
            0.0,
            1e-10,
            1.0000000000000001e-05,
            0.01,
            0.10000000000000001
        ];
        let expect = vec64![
            0.0,
            4.9999999998750022e-11,
            4.9999875000208323e-06,
            0.004987520807317688,
            0.048770575499285991
        ];
        let got = chi_square_cdf(&x, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-10);
    }

    #[test]
    fn chi2_cdf_cdf_negative() {
        let x = vec64![-10.0, -1.0, -0.10000000000000001, 0.0, 1.0];
        let expect = vec64![0.0, 0.0, 0.0, 0.0, 0.19874804309879915];
        let got = chi_square_cdf(&x, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn chi2_ppf_ppf_df_1() {
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
            1.5707971492624921e-06,
            0.00015708785790970184,
            0.015790774093431222,
            0.10153104426762156,
            0.454936423119572,
            1.3233036969314664,
            2.705543454095404,
            6.6348966010212145,
            10.827566170662733
        ];
        let got = chi_square_quantile(&q, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn chi2_ppf_ppf_df_2() {
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
            0.0020010006671670679,
            0.020100671707002873,
            0.21072103131565273,
            0.5753641449035618,
            1.386294361119891,
            2.7725887222397811,
            4.6051701859880918,
            9.2103403719761801,
            13.815510557964274
        ];
        let got = chi_square_quantile(&q, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn chi2_ppf_ppf_df_3() {
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
            0.024297585815692732,
            0.11483180189911707,
            0.58437437415518345,
            1.2125329030456686,
            2.3659738843753377,
            4.1083449356323118,
            6.2513886311703253,
            11.344866730144373,
            16.266236196238129
        ];
        let got = chi_square_quantile(&q, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn chi2_ppf_ppf_df_5() {
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
            0.2102126026292192,
            0.55429807672827724,
            1.6103079869623227,
            2.6746028094321637,
            4.3514601910955264,
            6.6256797638292468,
            9.2363568997811232,
            15.086272469388989,
            20.515005652432873
        ];
        let got = chi_square_quantile(&q, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn chi2_ppf_ppf_df_10() {
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
            1.4787434638356647,
            2.5582121601872063,
            4.8651820519253279,
            6.7372007719546421,
            9.3418177655919692,
            12.548861396889377,
            15.987179172105265,
            23.209251158954356,
            29.588298445074422
        ];
        let got = chi_square_quantile(&q, 10.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-12);
    }

    #[test]
    fn chi2_ppf_ppf_large_df() {
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
            61.917939206936623,
            70.064894925399784,
            82.358135812357148,
            90.133219746339321,
            99.334129235988456,
            109.1412410700806,
            118.49800381106212,
            135.80672317102676,
            149.44925277903886
        ];
        let got = chi_square_quantile(&q, 100.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 9.9999999999999998e-13);
    }

    #[test]
    fn chi2_ppf_ppf_extreme() {
        let q = vec64![
            1e-10,
            1.0000000000000001e-05,
            0.5,
            0.99999000000000005,
            0.99999999989999999
        ];
        let expect = vec64![
            5.2093976214344798e-07,
            1.1225825800018480e-03,
            2.3659738843753377e+00,
            2.5901749745671491e+01,
            4.9542155758766434e+01
        ];
        let got = chi_square_quantile(&q, 3.0, None, None).unwrap();
        println!("{}", got);
        assert_slice_close(&got, &expect, 1e-5);
    }

    #[test]
    fn chi2_ppf_ppf_boundaries() {
        let q = vec64![0.0, 1e-300, 0.5, 1.0, 1.0];
        let expect = vec64![
            0.0,
            3.2334077805831805e-120,
            4.3514601910955264,
            f64::INFINITY,
            f64::INFINITY
        ];
        let got = chi_square_quantile(&q, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }
}
