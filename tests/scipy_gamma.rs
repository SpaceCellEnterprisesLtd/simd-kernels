// This file is created by gen_scipy_tests.py and contains reference
// tests whose expected values are produced by SciPy at generation time.
//

// Each test compares our kernel outputs against SciPy with a per-test tolerance.
// NaN/Inf equality is handled by util::assert_slice_close.

mod util;
#[cfg(feature = "probability_distributions")]
mod scipy_gamma_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::gamma::{
        gamma_cdf, gamma_pdf, gamma_quantile,
    };

    // TODO[5]: Fix source for large_shape_2. re-running the Scipy file generation will ruin this file

    #[test]
    fn gamma_pdf_exponential() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.90483741803595952,
            0.60653065971263342,
            0.36787944117144233,
            0.1353352832366127,
            0.006737946999085467,
            4.5399929762484854e-05
        ];
        let got = gamma_pdf(&x, 1.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn gamma_pdf_erlang_2() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.090483741803595974,
            0.30326532985631671,
            0.36787944117144233,
            0.2706705664732254,
            0.033689734995427337,
            0.00045399929762484861
        ];
        let got = gamma_pdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn gamma_pdf_shape_less_1() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            2.0657661898691133,
            0.41510749742059477,
            0.10798193302637611,
            0.010333492677046023,
            1.6199821912178222e-05,
            5.2005637376543867e-10
        ];
        let got = gamma_pdf(&x, 0.5, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn gamma_pdf_large_shape() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            1.2385799798186382e-07,
            6.3378969976514042e-05,
            0.00078975346316749137,
            0.0076641550244050506,
            0.066800942890542642,
            0.087733684883925314
        ];
        let got = gamma_pdf(&x, 5.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn gamma_pdf_equal_params() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.10001045979203195,
            0.75306429050095058,
            0.67212542296616329,
            0.13385261753998334,
            0.00010324203316936629,
            1.2632791007934271e-10
        ];
        let got = gamma_pdf(&x, 3.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn gamma_pdf_very_small_shape() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.75549201382530706,
            0.11897044367129961,
            0.038669169440302374,
            0.0076233062353085416,
            0.00016638490099201526,
            6.0077867261998975e-07
        ];
        let got = gamma_pdf(&x, 0.10000000000000001, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn gamma_pdf_large_shape_2() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            1.5154304823524724e-97,
            1.2095689939909993e-63,
            2.5049897390996721e-49,
            1.9084763169562299e-35,
            1.4927267257775354e-18,
            3.8150942987200472e-08
        ];
        let got = gamma_pdf(&x, 50.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 9.9999999999999998e-13);
    }

    #[test]
    fn gamma_pdf_near_zero() {
        let x = vec64![
            0.0,
            1e-10,
            1.0000000000000001e-05,
            0.01,
            0.10000000000000001
        ];
        let expect = vec64![
            0.0,
            9.999999999000014e-11,
            9.9999000005000008e-06,
            0.009900498337491688,
            0.090483741803595974
        ];
        let got = gamma_pdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-10);
    }

    #[test]
    fn gamma_pdf_negative_values() {
        let x = vec64![-10.0, -1.0, -0.10000000000000001, 0.0, 1.0];
        let expect = vec64![0.0, 0.0, 0.0, 0.0, 0.36787944117144233];
        let got = gamma_pdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn gamma_cdf_cdf_exponential() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.095162581964040441,
            0.39346934028736652,
            0.63212055882855767,
            0.8646647167633873,
            0.99326205300091452,
            0.99995460007023751
        ];
        let got = gamma_cdf(&x, 1.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn gamma_cdf_cdf_erlang_2() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.0046788401604444738,
            0.090204010431049864,
            0.26424111765711528,
            0.59399415029016156,
            0.95957231800548726,
            0.99950060077261271
        ];
        let got = gamma_cdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn gamma_cdf_cdf_shape_less_1() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.47291074313446196,
            0.84270079294971512,
            0.95449973610364147,
            0.99532226501895271,
            0.99999225578356898,
            0.99999999974603715
        ];
        let got = gamma_cdf(&x, 0.5, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn gamma_cdf_cdf_large_shape() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            2.4979513360065075e-09,
            6.6117105610342441e-06,
            0.00017211562995584072,
            0.0036598468273437131,
            0.10882198108584877,
            0.55950671493478787
        ];
        let got = gamma_cdf(&x, 5.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn gamma_cdf_cdf_equal_params() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.0035994931830894716,
            0.19115316946194183,
            0.57680991887315658,
            0.93803119558334103,
            0.99996069155181555,
            0.99999999995498978
        ];
        let got = gamma_cdf(&x, 3.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn gamma_cdf_cdf_small_shape() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0];
        let expect = vec64![
            0.82755175958585037,
            0.94140244589013344,
            0.97587265627367215,
            0.99432617602018847,
            0.99985606103415325,
            0.99999944520142825
        ];
        let got = gamma_cdf(&x, 0.10000000000000001, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn gamma_cdf_cdf_near_zero() {
        let x = vec64![
            0.0,
            1e-10,
            1.0000000000000001e-05,
            0.01,
            0.10000000000000001
        ];
        let expect = vec64![
            0.0,
            4.9999999996666536e-21,
            4.9999666667916663e-11,
            4.9667913340265957e-05,
            0.0046788401604444738
        ];
        let got = gamma_cdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-10);
    }

    #[test]
    fn gamma_cdf_cdf_negative() {
        let x = vec64![-10.0, -1.0, -0.10000000000000001, 0.0, 1.0];
        let expect = vec64![0.0, 0.0, 0.0, 0.0, 0.26424111765711528];
        let got = gamma_cdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn gamma_ppf_ppf_exponential() {
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
            0.0010005003335835339,
            0.010050335853501437,
            0.10536051565782636,
            0.2876820724517809,
            0.69314718055994551,
            1.3862943611198906,
            2.3025850929940459,
            4.60517018598809,
            6.9077552789821368
        ];
        let got = gamma_quantile(&q, 1.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-12);
    }

    #[test]
    fn gamma_ppf_ppf_erlang_2() {
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
            0.045402017769489544,
            0.14855474025326595,
            0.53181160838961195,
            0.96127876311477711,
            1.6783469900166612,
            2.6926345288896951,
            3.8897201698674291,
            6.6383520679938108,
            9.2334134764515845
        ];
        let got = gamma_quantile(&q, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-12);
    }

    #[test]
    fn gamma_ppf_ppf_shape_less_1() {
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
            3.9269928731562303e-07,
            3.927196447742546e-05,
            0.0039476935233578054,
            0.025382761066905391,
            0.113734105779893,
            0.33082592423286661,
            0.67638586352385099,
            1.6587241502553036,
            2.7068915426656832
        ];
        let got = gamma_quantile(&q, 0.5, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn gamma_ppf_ppf_large_shape() {
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
        let got = gamma_quantile(&q, 5.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-12);
    }

    #[test]
    fn gamma_ppf_ppf_equal_params() {
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
            0.063511125856134387,
            0.14534838835943104,
            0.36735510941644045,
            0.575766472620173,
            0.89135343790785293,
            1.3068006867641868,
            1.7741067792780703,
            2.8019823049618209,
            3.7429574141375537
        ];
        let got = gamma_quantile(&q, 3.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-12);
    }

    #[test]
    fn gamma_ppf_ppf_extreme() {
        let q = vec64![
            1e-10,
            1.0000000000000001e-05,
            0.5,
            0.99999000000000005,
            0.99999999989999999
        ];
        let expect = vec64![
            1.414220229082974e-05,
            0.0044788163184577767,
            1.6783469900166612,
            14.236627712007888,
            26.333981519648543
        ];
        let got = gamma_quantile(&q, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-8);
    }

    #[test]
    fn gamma_ppf_ppf_small_shape() {
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
            6.0730483624078688e-31,
            6.0730483624079118e-21,
            6.0730483627432063e-11,
            5.7917132949696182e-07,
            0.00059339110446022842,
            0.035306358073558398,
            0.26615455373883701,
            1.5884778179295,
            3.3636770117187536
        ];
        let got = gamma_quantile(&q, 0.10000000000000001, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-13);
    }

    #[test]
    fn gamma_ppf_ppf_boundaries() {
        let q = vec64![0.0, 1e-300, 0.5, 1.0, 1.0];
        let expect = vec64![
            0.0,
            9.0856029641608548e-101,
            1.3370301568617795,
            f64::INFINITY,
            f64::INFINITY
        ];
        let got = gamma_quantile(&q, 3.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }
}
