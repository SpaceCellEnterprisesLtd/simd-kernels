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
mod scipy_normal_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::normal::{
        normal_cdf, normal_pdf, normal_quantile,
    };

    #[test]
    fn normal_pdf_standard() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.0044318484119380075,
            0.24197072451914337,
            0.3989422804014327,
            0.24197072451914337,
            0.0044318484119380075
        ];
        let got = normal_pdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_pdf_shifted_mean() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            1.4867195147342979e-06,
            0.0044318484119380075,
            0.053990966513188063,
            0.24197072451914337,
            0.24197072451914337
        ];
        let got = normal_pdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_pdf_different_variance() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.064758797832945872,
            0.17603266338214976,
            0.19947114020071635,
            0.17603266338214976,
            0.064758797832945872
        ];
        let got = normal_pdf(&x, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_pdf_small_variance() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            1.4736461348785476e-195,
            7.6945986267064195e-22,
            3.9894228040143269,
            7.6945986267064195e-22,
            1.4736461348785476e-195
        ];
        let got = normal_pdf(&x, 0.0, 0.10000000000000001, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_pdf_large_variance() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.038138781546052415,
            0.039695254747701178,
            0.039894228040143268,
            0.039695254747701178,
            0.038138781546052415
        ];
        let got = normal_pdf(&x, 0.0, 10.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_pdf_extreme_values() {
        let x = vec64![-50.0, -10.0, 0.0, 10.0, 50.0];
        let expect = vec64![
            0.0,
            7.6945986267064199e-23,
            0.3989422804014327,
            7.6945986267064199e-23,
            0.0
        ];
        let got = normal_pdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 9.9999999999999998e-13);
    }

    #[test]
    fn normal_pdf_near_mean() {
        let x = vec64![-0.01, -0.001, 0.0, 0.001, 0.01];
        let expect = vec64![
            0.39892233378608216,
            0.39894208093034239,
            0.3989422804014327,
            0.39894208093034239,
            0.39892233378608216
        ];
        let got = normal_pdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_pdf_negative_mean() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.00026766045152977074,
            1.0104542167073785e-14,
            1.538919725341284e-22,
            4.2927674713261209e-32,
            2.0523261455838072e-56
        ];
        let got = normal_pdf(&x, -5.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_pdf_very_small_variance() {
        let x = vec64![-0.10000000000000001, -0.01, 0.0, 0.01, 0.10000000000000001];
        let expect = vec64![
            7.6945986267064195e-21,
            24.197072451914337,
            39.894228040143268,
            24.197072451914337,
            7.6945986267064195e-21
        ];
        let got = normal_pdf(&x, 0.0, 0.01, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_pdf_multi_sd() {
        let x = vec64![-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0];
        let expect = vec64![
            6.0758828498232861e-09,
            0.00013383022576488537,
            0.053990966513188063,
            0.3989422804014327,
            0.053990966513188063,
            0.00013383022576488537,
            6.0758828498232861e-09
        ];
        let got = normal_pdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_cdf_cdf_standard() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.0013498980316300933,
            0.15865525393145707,
            0.5,
            0.84134474606854293,
            0.9986501019683699
        ];
        let got = normal_cdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_cdf_cdf_shifted() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            2.8665157187919328e-07,
            0.0013498980316300933,
            0.022750131948179195,
            0.15865525393145707,
            0.84134474606854293
        ];
        let got = normal_cdf(&x, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_cdf_cdf_scaled() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.066807201268858071,
            0.30853753872598688,
            0.5,
            0.69146246127401312,
            0.93319279873114191
        ];
        let got = normal_cdf(&x, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_cdf_cdf_negative_small_var() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.5,
            0.99996832875816688,
            0.9999999990134123,
            0.99999999999999933,
            1.0
        ];
        let got = normal_cdf(&x, -3.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_cdf_cdf_positive_large_var() {
        let x = vec64![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expect = vec64![
            0.0038303805675897365,
            0.022750131948179195,
            0.047790352272814703,
            0.091211219725867876,
            0.25249253754692291
        ];
        let got = normal_cdf(&x, 5.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_cdf_cdf_extreme() {
        let x = vec64![-50.0, -10.0, 0.0, 10.0, 50.0];
        let expect = vec64![0.0, 7.6198530241604696e-24, 0.5, 1.0, 1.0];
        let got = normal_cdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 9.9999999999999998e-13);
    }

    #[test]
    fn normal_cdf_cdf_near_median() {
        let x = vec64![-0.10000000000000001, -0.01, 0.0, 0.01, 0.10000000000000001];
        let expect = vec64![
            0.46017216272297101,
            0.4960106436853684,
            0.5,
            0.5039893563146316,
            0.53982783727702899
        ];
        let got = normal_cdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_cdf_cdf_small_var() {
        let x = vec64![-0.10000000000000001, -0.01, 0.0, 0.01, 0.10000000000000001];
        let expect = vec64![
            7.6198530241604696e-24,
            0.15865525393145707,
            0.5,
            0.84134474606854293,
            1.0
        ];
        let got = normal_cdf(&x, 0.0, 0.01, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_cdf_cdf_left_tail() {
        let x = vec64![-10.0, -8.0, -6.0, -4.0, -2.0];
        let expect = vec64![
            7.6198530241604696e-24,
            6.2209605742717405e-16,
            9.8658764503769458e-10,
            3.1671241833119863e-05,
            0.022750131948179195
        ];
        let got = normal_cdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 9.9999999999999998e-13);
    }

    #[test]
    fn normal_cdf_cdf_right_tail() {
        let x = vec64![2.0, 4.0, 6.0, 8.0, 10.0];
        let expect = vec64![
            0.97724986805182079,
            0.99996832875816688,
            0.9999999990134123,
            0.99999999999999933,
            1.0
        ];
        let got = normal_cdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 9.9999999999999998e-13);
    }

    #[test]
    fn normal_ppf_ppf_standard() {
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
            -3.0902323061678132,
            -2.3263478740408408,
            -1.2815515655446004,
            -0.67448975019608171,
            0.0,
            0.67448975019608171,
            1.2815515655446004,
            2.3263478740408408,
            3.0902323061678132
        ];
        let got = normal_quantile(&q, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_ppf_ppf_shifted() {
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
            -1.0902323061678132,
            -0.32634787404084076,
            0.71844843445539963,
            1.3255102498039184,
            2.0,
            2.6744897501960816,
            3.2815515655446004,
            4.3263478740408408,
            5.0902323061678132
        ];
        let got = normal_quantile(&q, 2.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_ppf_ppf_scaled() {
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
            -6.1804646123356264,
            -4.6526957480816815,
            -2.5631031310892007,
            -1.3489795003921634,
            0.0,
            1.3489795003921634,
            2.5631031310892007,
            4.6526957480816815,
            6.1804646123356264
        ];
        let got = normal_quantile(&q, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_ppf_ppf_negative_small_var() {
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
            -4.5451161530839066,
            -4.1631739370204208,
            -3.6407757827723,
            -3.3372448750980408,
            -3.0,
            -2.6627551249019592,
            -2.3592242172277,
            -1.8368260629795796,
            -1.4548838469160934
        ];
        let got = normal_quantile(&q, -3.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_ppf_ppf_positive_large_var() {
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
            -4.2706969185034396,
            -1.9790436221225223,
            1.1553453033661989,
            2.9765307494117548,
            5.0,
            7.0234692505882457,
            8.8446546966338015,
            11.979043622122521,
            14.27069691850344
        ];
        let got = normal_quantile(&q, 5.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_ppf_ppf_extreme_quantiles() {
        let q = vec64![
            1.0000000000000001e-15,
            1e-10,
            0.5,
            0.99999999989999999,
            0.999999999999999
        ];
        let expect = vec64![
            -7.9413453261709979,
            -6.3613409024040557,
            0.0,
            6.3613408896974217,
            7.9414444874159793
        ];
        let got = normal_quantile(&q, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-10);
    }

    #[test]
    fn normal_ppf_ppf_small_var() {
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
            -0.30902323061678133,
            -0.23263478740408408,
            -0.12815515655446005,
            -0.067448975019608171,
            0.0,
            0.067448975019608171,
            0.12815515655446005,
            0.23263478740408408,
            0.30902323061678133
        ];
        let got = normal_quantile(&q, 0.0, 0.10000000000000001, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_ppf_ppf_large_var() {
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
            -30.902323061678132,
            -23.263478740408409,
            -12.815515655446003,
            -6.7448975019608168,
            0.0,
            6.7448975019608168,
            12.815515655446003,
            23.263478740408409,
            30.902323061678132
        ];
        let got = normal_quantile(&q, 0.0, 10.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_ppf_ppf_near_median() {
        let q = vec64![0.48999999999999999, 0.495, 0.5, 0.505, 0.51000000000000001];
        let expect = vec64![
            -0.02506890825871106,
            -0.012533469508069276,
            0.0,
            0.012533469508069276,
            0.02506890825871106
        ];
        let got = normal_quantile(&q, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn normal_ppf_ppf_boundaries() {
        let q = vec64![0.0, 1e-300, 0.5, 1.0, 1.0];
        let expect = vec64![
            f64::NEG_INFINITY,
            -37.047096299361201,
            0.0,
            f64::INFINITY,
            f64::INFINITY
        ];
        let got = normal_quantile(&q, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-11);
    }
}
