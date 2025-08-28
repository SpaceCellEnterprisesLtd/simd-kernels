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
mod scipy_negative_binomial_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::neg_binomial::{
        neg_binomial_cdf, neg_binomial_pmf, neg_binomial_quantile,
    };
    // use simd_kernels::kernels::scientific::distributions::discrete::negative_binomial::{neg_binomial_pmf, neg_binomial_cdf, neg_binomial_quantile};

    #[test]
    fn neg_binomial_pmf_coin_flips() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25];
        let expect = vec64![
            0.03125,
            0.078125,
            0.1171875,
            0.13671874999999992,
            0.13671874999999997,
            0.12304687499999992,
            0.10253906249999994,
            0.08056640625,
            0.060424804687500028,
            0.043640136718750014,
            0.030548095703124986,
            0.0036964416503906233,
            0.00031667947769165007,
            2.2119842469692237e-05
        ];
        let got = neg_binomial_pmf(&k, 5, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_pmf_high_success() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25];
        let expect = vec64![
            0.51200000000000001,
            0.30720000000000003,
            0.12287999999999993,
            0.040959999999999983,
            0.012287999999999992,
            0.0034406399999999983,
            0.00091750399999999911,
            0.00023592959999999994,
            5.8982399999999912e-05,
            1.4417919999999984e-05,
            3.4603007999999937e-06,
            2.2817013759999928e-09,
            1.2401718067199951e-12,
            6.0301340835839632e-16
        ];
        let got = neg_binomial_pmf(&k, 3, 0.80000000000000004, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_pmf_low_success() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25];
        let expect = vec64![
            0.01,
            0.018000000000000006,
            0.024300000000000002,
            0.029159999999999995,
            0.032804999999999994,
            0.0354294,
            0.037200870000000011,
            0.038263751999999998,
            0.038742048899999999,
            0.038742048899999972,
            0.038354628410999986,
            0.032942581135143832,
            0.025531097464019544,
            0.018665347679988165
        ];
        let got = neg_binomial_pmf(&k, 2, 0.10000000000000001, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_pmf_single_success() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25];
        let expect = vec64![
            0.29999999999999988,
            0.21000000000000005,
            0.14700000000000002,
            0.10289999999999998,
            0.072030000000000025,
            0.050421000000000001,
            0.035294699999999977,
            0.024706290000000009,
            0.017294402999999986,
            0.012106082099999981,
            0.0084742574699999997,
            0.0014242684529828917,
            0.00023937679889283513,
            4.023205858991887e-05
        ];
        let got = neg_binomial_pmf(&k, 1, 0.29999999999999999, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_pmf_many_successes() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25];
        let expect = vec64![
            0.00010485760000000009,
            0.00062914559999999967,
            0.0020761804800000019,
            0.0049828331519999932,
            0.0097165246464000019,
            0.016323761405952021,
            0.024485642108927994,
            0.033580309177958401,
            0.042814894201896957,
            0.05137787304227636,
            0.058570775268195104,
            0.064463175477945583,
            0.038395125493161375,
            0.015636410564467254
        ];
        let got = neg_binomial_pmf(&k, 10, 0.40000000000000002, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn neg_binomial_pmf_near_certain() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25];
        let expect = vec64![
            0.97029899999999991,
            0.029108970000000026,
            0.00058217940000000106,
            9.7029900000000147e-06,
            1.4554485000000041e-07,
            2.0376279000000081e-09,
            2.7168372000000128e-11,
            3.4930764000000189e-13,
            4.3663455000000273e-15,
            5.3366445000000349e-17,
            6.4039734000000551e-19,
            1.319606640000018e-28,
            2.2413906900000385e-38,
            3.4057494900000837e-48
        ];
        let got = neg_binomial_pmf(&k, 3, 0.98999999999999999, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_pmf_quality_control() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25];
        let expect = vec64![
            3.1250000000000008e-07,
            1.4843750000000001e-06,
            4.2304687500000025e-06,
            9.377539062499996e-06,
            1.7817324218750001e-05,
            3.0467624414062504e-05,
            4.8240405322265625e-05,
            7.2016033659667985e-05,
            0.00010262284796502691,
            0.00014082135248534249,
            0.00018729239880550539,
            0.00056116150253100066,
            0.0011903973160978679,
            0.0020588374222552143
        ];
        let got = neg_binomial_pmf(&k, 5, 0.050000000000000003, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_pmf_balanced() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25];
        let expect = vec64![
            0.027993599999999993,
            0.078382080000000007,
            0.12541132799999999,
            0.15049359360000003,
            0.15049359359999989,
            0.13243436236800002,
            0.10594748989439999,
            0.078703849635840026,
            0.055092694745087989,
            0.036728463163392046,
            0.023506216424570934,
            0.0016310618380824498,
            7.0863156381631361e-05,
            2.3206098396940305e-06
        ];
        let got = neg_binomial_pmf(&k, 7, 0.59999999999999998, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_cdf_coin_flips_cdf() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25];
        let expect = vec64![
            0.03125,
            0.109375,
            0.2265625,
            0.36328125,
            0.5,
            0.623046875,
            0.7255859375,
            0.80615234375,
            0.8665771484375,
            0.91021728515625,
            0.940765380859375,
            0.99409103393554688,
            0.99954473972320557,
            0.99997026193886995
        ];
        let got = neg_binomial_cdf(&k, 5, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn neg_binomial_cdf_high_success_cdf() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25];
        let expect = vec64![
            0.51200000000000001,
            0.81920000000000004,
            0.94208000000000003,
            0.98304000000000002,
            0.99532799999999999,
            0.99876863999999999,
            0.999686144,
            0.99992207359999996,
            0.99998105599999998,
            0.99999547391999999,
            0.9999989342208,
            0.99999999933913497,
            0.99999999999965261,
            0.99999999999999989
        ];
        let got = neg_binomial_cdf(&k, 3, 0.80000000000000004, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_cdf_low_success_cdf() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25];
        let expect = vec64![
            0.010000000000000002,
            0.028000000000000008,
            0.052300000000000013,
            0.081460000000000005,
            0.11426500000000002,
            0.14969440000000003,
            0.18689527000000009,
            0.22515902199999999,
            0.26390107089999998,
            0.30264311980000003,
            0.3409977482110001,
            0.51821475089852176,
            0.66080113369231142,
            0.76740105198783959
        ];
        let got = neg_binomial_cdf(&k, 2, 0.10000000000000001, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn neg_binomial_cdf_single_success_cdf() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25];
        let expect = vec64![
            0.29999999999999999,
            0.51000000000000001,
            0.65699999999999992,
            0.75990000000000002,
            0.83192999999999995,
            0.88235099999999989,
            0.91764570000000001,
            0.94235199000000003,
            0.95964639299999999,
            0.97175247509999996,
            0.98022673257000004,
            0.99667670694303989,
            0.99944145413591667,
            0.99990612519662347
        ];
        let got = neg_binomial_cdf(&k, 1, 0.29999999999999999, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_cdf_many_successes_cdf() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25];
        let expect = vec64![
            0.00010485760000000006,
            0.00073400320000000028,
            0.0028101836800000003,
            0.0077930168320000017,
            0.017509541478400004,
            0.033833302884352018,
            0.058318944993280025,
            0.091899254171238412,
            0.13471414837313533,
            0.18609202141541176,
            0.24466279668360685,
            0.57538298232899254,
            0.82371351525598757,
            0.94247329259899093
        ];
        let got = neg_binomial_cdf(&k, 10, 0.40000000000000002, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn neg_binomial_cdf_near_certain_cdf() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25];
        let expect = vec64![
            0.97029900000000002,
            0.99940797000000003,
            0.99999014939999997,
            0.99999985239,
            0.99999999793484995,
            0.9999999999724779,
            0.99999999999964628,
            0.99999999999999556,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        ];
        let got = neg_binomial_cdf(&k, 3, 0.98999999999999999, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_cdf_quality_control_cdf() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25];
        let expect = vec64![
            3.1250000000000008e-07,
            1.7968750000000005e-06,
            6.0273437499999995e-06,
            1.54048828125e-05,
            3.3222207031249994e-05,
            6.3689831445312494e-05,
            0.0001119302367675781,
            0.00018394627042724596,
            0.00028656911839227285,
            0.00042739047087761521,
            0.00061468286968312052,
            0.0025739403346522792,
            0.0071649479025865772,
            0.015635510128533071
        ];
        let got = neg_binomial_cdf(&k, 5, 0.050000000000000003, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_cdf_balanced_cdf() {
        let k = vec64![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25];
        let expect = vec64![
            0.027993599999999987,
            0.10637567999999997,
            0.23178700799999996,
            0.38228060159999988,
            0.53277419520000002,
            0.66520855756800013,
            0.77115604746239996,
            0.84985989709824006,
            0.90495259184332799,
            0.94168105500671995,
            0.96518727143129091,
            0.99807779691119469,
            0.99992642320951453,
            0.99999777693772329
        ];
        let got = neg_binomial_cdf(&k, 7, 0.59999999999999998, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn neg_binomial_cdf_extended_cdf() {
        let k = vec64![0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
        let expect = vec64![
            0.00032000000000000008,
            0.03279349760000002,
            0.16423372393676816,
            0.3703517360973313,
            0.57932569074786777,
            0.74476674525968067,
            0.85650829030716524,
            0.92408550455010552,
            0.96176400169974519,
            0.98150398493979063,
            0.99134917085216323
        ];
        let got = neg_binomial_cdf(&k, 5, 0.20000000000000001, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn neg_binomial_ppf_coin_flips_ppf() {
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
        let expect = vec64![0.0, 1.0, 1.0, 3.0, 4.0, 7.0, 9.0, 11.0, 14.0];
        let got = neg_binomial_quantile(&q, 5, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_ppf_high_success_ppf() {
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
        let expect = vec64![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let got = neg_binomial_quantile(&q, 3, 0.80000000000000004, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_ppf_low_success_ppf() {
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
        let expect = vec64![0.0, 2.0, 4.0, 8.0, 15.0, 25.0, 36.0, 44.0, 62.0];
        let got = neg_binomial_quantile(&q, 2, 0.10000000000000001, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_ppf_single_success_ppf() {
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
        let expect = vec64![0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 6.0, 8.0, 12.0];
        let got = neg_binomial_quantile(&q, 1, 0.29999999999999999, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_ppf_many_successes_ppf() {
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
        let expect = vec64![4.0, 6.0, 8.0, 11.0, 14.0, 19.0, 23.0, 26.0, 32.0];
        let got = neg_binomial_quantile(&q, 10, 0.40000000000000002, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_ppf_near_certain_ppf() {
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
        let expect = vec64![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let got = neg_binomial_quantile(&q, 3, 0.98999999999999999, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_ppf_quality_control_ppf() {
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
        let expect = vec64![22.0, 36.0, 45.0, 63.0, 89.0, 120.0, 153.0, 176.0, 224.0];
        let got = neg_binomial_quantile(&q, 5, 0.050000000000000003, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_ppf_balanced_ppf() {
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
        let expect = vec64![0.0, 1.0, 1.0, 3.0, 4.0, 6.0, 8.0, 10.0, 13.0];
        let got = neg_binomial_quantile(&q, 7, 0.59999999999999998, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn neg_binomial_pmf_negbinom_standard() {
        let k = vec64![0, 1, 3, 5, 10];
        let expect = vec64![
            0.010239999999999996,
            0.030719999999999997,
            0.077414399999999994,
            0.10032906239999996,
            0.061979281588224036
        ];
        let got = neg_binomial_pmf(&k, 5, 0.40000000000000002, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn neg_binomial_pmf_negbinom_large_n() {
        let k = vec64![0, 1, 3, 5, 10];
        let expect = vec64![
            5.9048999999999966e-06,
            4.1334299999999961e-05,
            0.00044558375400000023,
            0.0019868579590859962,
            0.015408540450042514
        ];
        let got = neg_binomial_pmf(&k, 10, 0.29999999999999999, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn neg_binomial_pmf_negbinom_small_n() {
        let k = vec64![0, 1, 3, 5, 10];
        let expect = vec64![
            0.21599999999999994,
            0.25919999999999993,
            0.13824,
            0.04644863999999993,
            0.0014948499456000008
        ];
        let got = neg_binomial_pmf(&k, 3, 0.59999999999999998, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn neg_binomial_pmf_negbinom_low_p() {
        let k = vec64![0, 1, 3, 5, 10];
        let expect = vec64![
            9.9999999999999974e-06,
            4.4999999999999996e-05,
            0.00025514999999999983,
            0.00074401740000000038,
            0.0034902711854010032
        ];
        let got = neg_binomial_pmf(&k, 5, 0.10000000000000001, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn neg_binomial_cdf_negbinom_cdf_standard() {
        let k = vec64![0, 1, 3, 5, 10];
        let expect = vec64![
            0.010240000000000003,
            0.04096000000000001,
            0.17367040000000003,
            0.36689674240000009,
            0.78272229434982399
        ];
        let got = neg_binomial_cdf(&k, 5, 0.40000000000000002, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn neg_binomial_cdf_negbinom_cdf_large_n() {
        let k = vec64![0, 1, 3, 5, 10];
        let expect = vec64![
            5.9048999999999975e-06,
            4.723919999999998e-05,
            0.00065196000899999957,
            0.0036525210084359973,
            0.047961897331343421
        ];
        let got = neg_binomial_cdf(&k, 10, 0.29999999999999999, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn neg_binomial_cdf_negbinom_cdf_small_n() {
        let k = vec64![0, 1, 3, 5, 10];
        let expect = vec64![
            0.21599999999999997,
            0.47520000000000001,
            0.82079999999999997,
            0.95019264000000003,
            0.99868466626560004
        ];
        let got = neg_binomial_cdf(&k, 3, 0.59999999999999998, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn neg_binomial_ppf_negbinom_ppf_standard() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![0.0, 3.0, 7.0, 13.0, 20.0];
        let got = neg_binomial_quantile(&q, 5, 0.40000000000000002, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn neg_binomial_ppf_negbinom_ppf_large_n() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![7.0, 13.0, 22.0, 35.0, 48.0];
        let got = neg_binomial_quantile(&q, 10, 0.29999999999999999, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn neg_binomial_ppf_negbinom_ppf_small_n() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![0.0, 0.0, 2.0, 4.0, 8.0];
        let got = neg_binomial_quantile(&q, 3, 0.59999999999999998, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }
}
