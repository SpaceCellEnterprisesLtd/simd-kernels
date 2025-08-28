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
mod scipy_beta_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::beta::{
        beta_cdf, beta_pdf, beta_quantile,
    };

    #[test]
    fn beta_pdf_standard_symmetric() {
        let x = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let expect = vec64![
            0.54000000000000015,
            1.2600000000000002,
            1.5000000000000007,
            1.2600000000000005,
            0.53999999999999992
        ];
        let got = beta_pdf(&x, 2.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn beta_pdf_u_shaped() {
        let x = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let expect = vec64![
            1.0610329539459686,
            0.69460911804285663,
            0.63661977236758138,
            0.69460911804285663,
            1.0610329539459691
        ];
        let got = beta_pdf(&x, 0.5, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn beta_pdf_left_skewed() {
        let x = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let expect = vec64![
            0.0026999999999999997,
            0.17009999999999992,
            0.9375,
            2.1608999999999998,
            1.9682999999999995
        ];
        let got = beta_pdf(&x, 5.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn beta_pdf_right_skewed() {
        let x = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let expect = vec64![
            1.9682999999999997,
            2.1608999999999994,
            0.93749999999999989,
            0.17009999999999995,
            0.0026999999999999993
        ];
        let got = beta_pdf(&x, 2.0, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn beta_pdf_boundaries() {
        let x = vec64![0.0, 1e-10, 0.5, 0.99999999989999999, 1.0];
        let expect = vec64![
            0.0,
            1.1999999997599999e-09,
            1.5000000000000004,
            1.2000001984568983e-19,
            0.0
        ];
        let got = beta_pdf(&x, 2.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-10);
    }

    #[test]
    fn beta_pdf_large_params() {
        let x = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let expect = vec64![
            1.3361640091440897e-43,
            3.5938629673509188e-07,
            11.269695801851285,
            3.5938629673510554e-07,
            1.3361640091440329e-43
        ];
        let got = beta_pdf(&x, 100.0, 100.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-12);
    }

    #[test]
    fn beta_pdf_small_a() {
        let x = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let expect = vec64![
            0.78638495237703876,
            0.22755249197840463,
            0.10263362906904881,
            0.045491023055648547,
            0.012094124266964839
        ];
        let got = beta_pdf(&x, 0.1, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn beta_pdf_small_b() {
        let x = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let expect = vec64![
            0.012094124266964846,
            0.045491023055648547,
            0.10263362906904883,
            0.22755249197840474,
            0.78638495237703931
        ];
        let got = beta_pdf(&x, 2.0, 0.1, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn beta_pdf_uniform() {
        let x = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let expect = vec64![
            0.99999999999999944,
            0.99999999999999956,
            0.99999999999999956,
            0.99999999999999978,
            0.99999999999999944
        ];
        let got = beta_pdf(&x, 1.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn beta_pdf_domain_violations() {
        let x = vec64![-0.5, -0.10000000000000001, 0.5, 1.1000000000000001, 1.5];
        let expect = vec64![0.0, 0.0, 1.5000000000000004, 0.0, 0.0];
        let got = beta_pdf(&x, 2.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn beta_cdf_cdf_symmetric() {
        let x = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let expect = vec64![
            0.028000000000000008,
            0.21599999999999994,
            0.5,
            0.78399999999999992,
            0.97199999999999998
        ];
        let got = beta_cdf(&x, 2.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn beta_cdf_cdf_u_shaped() {
        let x = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let expect = vec64![
            0.20483276469913345,
            0.36901011956554536,
            0.50000000000000011,
            0.63098988043445459,
            0.79516723530086653
        ];
        let got = beta_cdf(&x, 0.5, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn beta_cdf_cdf_left_skewed() {
        let x = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let expect = vec64![
            5.5000000000000016e-05,
            0.010934999999999997,
            0.109375,
            0.42017499999999991,
            0.88573500000000005
        ];
        let got = beta_cdf(&x, 5.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn beta_cdf_cdf_right_skewed() {
        let x = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let expect = vec64![
            0.11426500000000002,
            0.57982500000000026,
            0.890625,
            0.98906499999999997,
            0.99994499999999997
        ];
        let got = beta_cdf(&x, 2.0, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn beta_cdf_cdf_uniform() {
        let x = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let expect = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let got = beta_cdf(&x, 1.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn beta_cdf_cdf_boundaries() {
        let x = vec64![0.0, 1e-10, 0.5, 0.99999999989999999, 1.0];
        let expect = vec64![0.0, 5.9999999992000006e-20, 0.6875, 1.0, 1.0];
        let got = beta_cdf(&x, 2.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-10);
    }

    #[test]
    fn beta_cdf_cdf_large_params() {
        let x = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let expect = vec64![
            1.4990328239114214e-46,
            1.8411141115800243e-09,
            0.49999999999999939,
            0.99999999815888585,
            1.0
        ];
        let got = beta_cdf(&x, 100.0, 100.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-12);
    }

    #[test]
    fn beta_cdf_cdf_domain_violations() {
        let x = vec64![-0.5, -0.10000000000000001, 0.5, 1.1000000000000001, 1.5];
        let expect = vec64![0.0, 0.0, 0.6875, 1.0, 1.0];
        let got = beta_cdf(&x, 2.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn beta_cdf_cdf_extreme_right_skew() {
        let x = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let expect = vec64![
            0.97614394925562586,
            0.99905785019837601,
            0.99997770252754181,
            0.9999998962997414,
            0.99999999999856448
        ];
        let got = beta_cdf(&x, 0.1, 10.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn beta_cdf_cdf_extreme_left_skew() {
        let x = vec64![
            0.10000000000000001,
            0.29999999999999999,
            0.5,
            0.69999999999999996,
            0.90000000000000002
        ];
        let expect = vec64![
            1.43536789168095e-12,
            1.0370025857993869e-07,
            2.2297472458379084e-05,
            0.00094214980162418856,
            0.023856050744374135
        ];
        let got = beta_cdf(&x, 10.0, 0.1, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn beta_ppf_ppf_symmetric() {
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
            0.058903135778195254,
            0.19580010565909173,
            0.3263518223330697,
            0.5,
            0.6736481776669303,
            0.80419989434090833,
            0.94109686422180472
        ];
        let got = beta_quantile(&q, 2.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn beta_ppf_ppf_u_shaped() {
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
            0.00024671981713422146,
            0.024471741852423214,
            0.14644660940672624,
            0.49999999999999989,
            0.85355339059327373,
            0.97552825814757682,
            0.9997532801828658
        ];
        let got = beta_quantile(&q, 0.5, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn beta_ppf_ppf_left_skewed() {
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
            0.29431367168029249,
            0.48968369344850837,
            0.61052051479927549,
            0.73555001670433995,
            0.83883708320967343,
            0.9074047410868713,
            0.97323680885724495
        ];
        let got = beta_quantile(&q, 5.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn beta_ppf_ppf_right_skewed() {
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
            0.026763191142755053,
            0.092595258913128725,
            0.16116291679032652,
            0.26444998329566005,
            0.38947948520072451,
            0.51031630655149174,
            0.70568632831970746
        ];
        let got = beta_quantile(&q, 2.0, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn beta_ppf_ppf_uniform() {
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
            0.01,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.98999999999999999
        ];
        let got = beta_quantile(&q, 1.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn beta_ppf_ppf_extreme_quantiles() {
        let q = vec64![
            1e-10,
            1.0000000000000001e-05,
            0.5,
            0.99999000000000005,
            0.99999999989999999
        ];
        let expect = vec64![
            4.0824940158083332e-06,
            0.0012921074164783165,
            0.38572756813238951,
            0.98638138649426754,
            0.99970757683872491
        ];
        let got = beta_quantile(&q, 2.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-8);
    }

    #[test]
    fn beta_ppf_ppf_large_params() {
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
            0.41820388514317364,
            0.45472688523155341,
            0.47613697029945945,
            0.50000000000000011,
            0.52386302970054055,
            0.54527311476844664,
            0.58179611485682636
        ];
        let got = beta_quantile(&q, 100.0, 100.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-12);
    }

    #[test]
    fn beta_ppf_ppf_small_params() {
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
            8.8692806555502085e-18,
            8.8692800119345832e-08,
            0.00084525553284712954,
            0.50000000000000011,
            0.99915474446715291,
            0.9999999113071999,
            1.0
        ];
        let got = beta_quantile(&q, 0.1, 0.1, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn beta_ppf_ppf_invalid_quantiles() {
        let q = vec64![-0.10000000000000001, 0.0, 0.5, 1.0, 1.1000000000000001];
        let expect = vec64![f64::NAN, 0.0, 0.38572756813238951, 1.0, f64::NAN];
        let got = beta_quantile(&q, 2.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn beta_ppf_ppf_extreme_asymmetry() {
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
            0.91201083935590976,
            0.954992586021436,
            0.97265494741228553,
            0.9862327044933592,
            0.99426287904640476,
            0.9978950082958632,
            0.99979901348342659
        ];
        let got = beta_quantile(&q, 50.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }
}
