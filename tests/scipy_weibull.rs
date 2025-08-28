// AUTO-GENERATED FILE. DO NOT EDIT BY HAND.
// Generated from SciPy 1.16.1 on 2025-08-21T22:25:23Z.
// Distribution: Weibull
//
// This file is created by gen_scipy_tests.py and contains reference
// tests whose expected values are produced by SciPy at generation time.
//

// Each test compares our kernel outputs against SciPy with a per-test tolerance.
// NaN/Inf equality is handled by util::assert_slice_close.

mod util;

#[cfg(feature = "probability_distributions")]
mod scipy_weibull_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::weibull::{
        weibull_cdf, weibull_pdf, weibull_quantile,
    };

    // ---- PDF ----

    #[test]
    fn weibull_pdf_weibull_decreasing() {
        let x = vec64![0.1, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            1.152481680041995,
            0.34865221527635115,
            0.18393972058572117,
            0.08595474576918094,
            0.023898630707079163
        ];
        let got = weibull_pdf(&x, 0.5, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn weibull_pdf_weibull_exponential() {
        let x = vec64![0.1, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.9048374180359595,
            0.6065306597126334,
            0.36787944117144233,
            0.1353352832366127,
            0.006737946999085467
        ];
        let got = weibull_pdf(&x, 1.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn weibull_pdf_weibull_rayleigh() {
        let x = vec64![0.1, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.19800996674983362,
            0.7788007830714049,
            0.7357588823428847,
            0.07326255555493671,
            1.3887943864964022e-10
        ];
        let got = weibull_pdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn weibull_pdf_weibull_normal_like() {
        let x = vec64![0.1, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.003749531279295655,
            0.09229654096925705,
            0.3309363384692233,
            0.5518191617571635,
            1.535041059928887e-06
        ];
        let got = weibull_pdf(&x, 3.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn weibull_pdf_weibull_sharp() {
        let x = vec64![0.1, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.000499995000025,
            0.30288538577385754,
            1.8393972058572117,
            1.013133243927534e-12,
            0.0
        ];
        let got = weibull_pdf(&x, 5.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    // identities: Weibull(c=1) <-> Exponential, Weibull(c=2) <-> Rayleigh

    #[test]
    fn weibull_pdf_weibull_expon_identity_weibull() {
        let x = vec64![0.5, 1.0, 2.0, 4.0, 8.0];
        let expect = vec64![
            0.6065306597126334,
            0.36787944117144233,
            0.1353352832366127,
            0.01831563888873418,
            0.00033546262790251185
        ];
        let got = weibull_pdf(&x, 1.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn weibull_pdf_weibull_rayleigh_identity_weibull() {
        let x = vec64![0.5, 1.0, 2.0, 4.0, 8.0];
        let expect = vec64![
            0.7788007830714049,
            0.7357588823428847,
            0.07326255555493671,
            9.002813977540729e-07,
            2.5660974248778207e-27
        ];
        let got = weibull_pdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    // ---- CDF ----

    #[test]
    fn weibull_cdf_weibull_cdf_decreasing() {
        let x = vec64![0.1, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.2711065858899754,
            0.5069313086047602,
            0.6321205588285577,
            0.7568832655657858,
            0.8931220743396142
        ];
        let got = weibull_cdf(&x, 0.5, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn weibull_cdf_weibull_cdf_exponential() {
        let x = vec64![0.1, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.09516258196404044,
            0.3934693402873666,
            0.6321205588285577,
            0.8646647167633873,
            0.9932620530009145
        ];
        let got = weibull_cdf(&x, 1.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn weibull_cdf_weibull_cdf_rayleigh() {
        let x = vec64![0.1, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.009950166250831947,
            0.22119921692859515,
            0.6321205588285577,
            0.9816843611112658,
            0.9999999999861121
        ];
        let got = weibull_cdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn weibull_cdf_weibull_cdf_normal_like() {
        let x = vec64![0.1, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.0001249921878255107,
            0.015503562994591595,
            0.1175030974154046,
            0.6321205588285577,
            0.999999836262287
        ];
        let got = weibull_cdf(&x, 3.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn weibull_cdf_weibull_cdf_sharp() {
        let x = vec64![0.1, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            9.999950000166668e-06,
            0.030766765523655912,
            0.6321205588285577,
            0.9999999999999873,
            1.0
        ];
        let got = weibull_cdf(&x, 5.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    // ---- PPF ----

    #[test]
    fn weibull_ppf_weibull_ppf_decreasing() {
        let q = vec64![0.01, 0.1, 0.5, 0.9, 0.99];
        let expect = vec64![
            0.00010100925076817656,
            0.011100838259683063,
            0.4804530139182014,
            5.301898110478399,
            21.207592441913587
        ];
        let got = weibull_quantile(&q, 0.5, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn weibull_ppf_weibull_ppf_exponential() {
        let q = vec64![0.01, 0.1, 0.5, 0.9, 0.99];
        let expect = vec64![
            0.010050335853501442,
            0.10536051565782631,
            0.6931471805599453,
            2.302585092994046,
            4.605170185988091
        ];
        let got = weibull_quantile(&q, 1.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn weibull_ppf_weibull_ppf_rayleigh() {
        let q = vec64![0.01, 0.1, 0.5, 0.9, 0.99];
        let expect = vec64![
            0.100251363349839,
            0.3245928459745013,
            0.8325546111576977,
            1.5174271293851465,
            2.145966026289347
        ];
        let got = weibull_quantile(&q, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn weibull_ppf_weibull_ppf_normal_like() {
        let q = vec64![0.01, 0.1, 0.5, 0.9, 0.99];
        let expect = vec64![
            0.4316086970718717,
            0.944617437139326,
            1.7699940890010355,
            2.6410009569073707,
            3.3274526984000987
        ];
        let got = weibull_quantile(&q, 3.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn weibull_ppf_weibull_ppf_sharp() {
        let q = vec64![0.01, 0.1, 0.5, 0.9, 0.99];
        let expect = vec64![
            0.3985071473196205,
            0.6375813096953713,
            0.9293195901316053,
            1.1815256050421783,
            1.3572165188988268
        ];
        let got = weibull_quantile(&q, 5.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }
}
