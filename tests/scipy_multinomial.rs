// AUTO-GENERATED FILE. DO NOT EDIT BY HAND.
// Generated from SciPy 1.16.1 on 2025-08-21T22:25:23Z.
// Distribution: Multinomial
//
// This file is created by gen_scipy_tests.py and contains reference
// tests whose expected values are produced by SciPy at generation time.
//

// Each test compares our kernel outputs against SciPy with a per-test tolerance.
// NaN/Inf equality is handled by util::assert_slice_close.

mod util;

#[cfg(feature = "probability_distributions")]
mod scipy_multinomial_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::multinomial::multinomial_pmf;

    #[test]
    fn multinomial_pmf_fair_3cat() {
        // rows: [1,1,4], [2,2,2], [3,1,2], [0,3,3], [1,0,5]
        let xs = vec64![1u64, 1, 4, 2, 2, 2, 3, 1, 2, 0, 3, 3, 1, 0, 5];
        let probs = vec64![
            0.33333333333333331,
            0.33333333333333331,
            0.33333333333333331
        ];
        let expect = vec64![
            0.041152263374485541,
            0.12345679012345677,
            0.082304526748971235,
            0.027434842249657095,
            0.0082304526748971148
        ];
        let got = multinomial_pmf(&xs, 6, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_biased_3cat() {
        let xs = vec64![1u64, 1, 4, 2, 2, 2, 3, 1, 2, 0, 3, 3, 1, 0, 5];
        let probs = vec64![0.5, 0.29999999999999999, 0.20000000000000001];
        let expect = vec64![
            0.0071999999999999998,
            0.080999999999999975,
            0.090000000000000011,
            0.0043200000000000009,
            0.00096000000000000143
        ];
        let got = multinomial_pmf(&xs, 6, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_fair_4cat() {
        let xs = vec64![
            1u64, 1, 1, 2, 2, 1, 1, 1, 0, 2, 2, 1, 1, 0, 0, 4, 0, 0, 1, 4
        ];
        let probs = vec64![0.25, 0.25, 0.25, 0.25];
        let expect = vec64![
            0.058593750000000035,
            0.058593749999999986,
            0.029296875000000017,
            0.0048828124999999965,
            0.0048828124999999965
        ];
        let got = multinomial_pmf(&xs, 5, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_skewed_4cat() {
        let xs = vec64![
            1u64, 1, 1, 2, 2, 1, 1, 1, 0, 2, 2, 1, 1, 0, 0, 4, 0, 0, 1, 4
        ];
        let probs = vec64![
            0.69999999999999996,
            0.20000000000000001,
            0.050000000000000003,
            0.050000000000000003
        ];
        let expect = vec64![
            0.0010500000000000015,
            0.014699999999999996,
            0.00015000000000000031,
            2.1874999999999976e-05,
            1.5624999999999977e-06
        ];
        let got = multinomial_pmf(&xs, 5, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_binary() {
        let xs = vec64![0u64, 3, 1, 2, 2, 1, 3, 0];
        let probs = vec64![0.40000000000000002, 0.59999999999999998];
        let expect = vec64![
            0.21600000000000003,
            0.43199999999999988,
            0.28799999999999998,
            0.064000000000000015
        ];
        let got = multinomial_pmf(&xs, 3, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_large_trials() {
        let xs = vec64![15u64, 10, 25, 20, 15, 15, 10, 20, 20, 25, 15, 10];
        let probs = vec64![
            0.29999999999999999,
            0.29999999999999999,
            0.40000000000000002
        ];
        let expect = vec64![
            0.0039418282788887789,
            0.0039273066996958709,
            0.0032055316848133911,
            5.2676468513668347e-05
        ];
        let got = multinomial_pmf(&xs, 50, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_equal_5cat() {
        let xs = vec64![
            2u64, 2, 2, 2, 2, 3, 2, 2, 2, 1, 1, 3, 2, 2, 2, 0, 0, 5, 3, 2
        ];
        let probs = vec64![
            0.20000000000000001,
            0.20000000000000001,
            0.20000000000000001,
            0.20000000000000001,
            0.20000000000000001
        ];
        let expect = vec64![
            0.011612160000000028,
            0.0077414400000000239,
            0.0077414400000000239,
            0.0002580480000000002
        ];
        let got = multinomial_pmf(&xs, 10, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_dominant_cat() {
        let xs = vec64![
            2u64, 2, 2, 2, 2, 3, 2, 2, 2, 1, 1, 3, 2, 2, 2, 0, 0, 5, 3, 2
        ];
        let probs = vec64![
            0.84999999999999998,
            0.050000000000000003,
            0.050000000000000003,
            0.029999999999999999,
            0.02
        ];
        let expect = vec64![
            1.8434587500000021e-07,
            5.2231331249999913e-06,
            7.2292500000000315e-09,
            8.5049999999999505e-12
        ];
        let got = multinomial_pmf(&xs, 10, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_small_trials() {
        let xs = vec64![
            0u64, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0
        ];
        let probs = vec64![
            0.29999999999999999,
            0.20000000000000001,
            0.20000000000000001,
            0.14999999999999999,
            0.14999999999999999
        ];
        let expect = vec64![
            0.059999999999999998,
            0.12000000000000001,
            0.045000000000000005,
            0.089999999999999983
        ];
        let got = multinomial_pmf(&xs, 2, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_single_trial() {
        let xs = vec64![1u64, 0, 0, 0, 1, 0, 0, 0, 1];
        let probs = vec64![0.5, 0.29999999999999999, 0.20000000000000001];
        let expect = vec64![0.5, 0.29999999999999999, 0.20000000000000001];
        let got = multinomial_pmf(&xs, 1, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_fair_dice() {
        let xs = vec64![1u64, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let probs = vec64![
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666
        ];
        let expect = vec64![
            0.01543209876543212,
            0.01543209876543212,
            0.01543209876543212
        ];
        let got = multinomial_pmf(&xs, 6, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_market_stack() {
        let xs = vec64![8u64, 6, 4, 2, 5, 5, 5, 5, 17, 1, 1, 1];
        let probs = vec64![
            0.40000000000000002,
            0.29999999999999999,
            0.20000000000000001,
            0.10000000000000001
        ];
        let expect = vec64![
            0.01334620530199755,
            0.00093423437113982822,
            7.0506183131135741e-06
        ];
        let got = multinomial_pmf(&xs, 20, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_survey_4point() {
        let xs = vec64![25u64, 25, 25, 25, 25, 25, 25, 25, 97, 1, 1, 1];
        let probs = vec64![0.25, 0.25, 0.25, 0.25];
        let expect = vec64![
            0.0010032791954709059,
            0.0010032791954709059,
            6.0375694225804617e-55
        ];
        let got = multinomial_pmf(&xs, 100, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn multinomial_pmf_product_quality() {
        let xs = vec64![10u64, 3, 1, 1, 4, 4, 4, 3, 12, 1, 1, 1];
        let probs = vec64![
            0.59999999999999998,
            0.25,
            0.10000000000000001,
            0.050000000000000003
        ];
        let expect = vec64![
            0.028371863520000037,
            9.976763671875046e-05,
            0.0074282697216000127
        ];
        let got = multinomial_pmf(&xs, 15, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_risk_categories() {
        let xs = vec64![4u64, 1, 1, 1, 1, 2, 2, 2, 1, 1, 4, 1, 1, 1, 1];
        let probs = vec64![
            0.5,
            0.20000000000000001,
            0.14999999999999999,
            0.10000000000000001,
            0.050000000000000003
        ];
        let expect = vec64![
            0.015750000000000014,
            0.0056700000000000066,
            0.015750000000000014
        ];
        let got = multinomial_pmf(&xs, 8, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_seasonal_sales() {
        let xs = vec64![5u64, 3, 2, 2, 3, 3, 3, 3, 9, 1, 1, 1];
        let probs = vec64![
            0.29999999999999999,
            0.29999999999999999,
            0.20000000000000001,
            0.20000000000000001
        ];
        let expect = vec64![
            0.017459608319999941,
            0.017244057599999973,
            0.00031177871999999991
        ];
        let got = multinomial_pmf(&xs, 12, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_customer_satisfaction() {
        let xs = vec64![10u64, 6, 5, 3, 1, 5, 5, 5, 5, 5, 21, 1, 1, 1, 1];
        let probs = vec64![
            0.34999999999999998,
            0.25,
            0.20000000000000001,
            0.14999999999999999,
            0.050000000000000003
        ];
        let expect = vec64![
            0.0029986923274158308,
            2.4279291480617928e-05,
            3.0322287857952394e-08
        ];
        let got = multinomial_pmf(&xs, 25, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_website_sections() {
        let xs = vec64![13u64, 9, 4, 3, 1, 6, 6, 6, 6, 6, 26, 1, 1, 1, 1];
        let probs = vec64![
            0.40000000000000002,
            0.29999999999999999,
            0.14999999999999999,
            0.10000000000000001,
            0.050000000000000003
        ];
        let expect = vec64![
            0.0027255895399568436,
            7.2853873849782669e-07,
            6.6647419805566814e-09
        ];
        let got = multinomial_pmf(&xs, 30, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_email_responses() {
        let xs = vec64![10u64, 4, 2, 1, 1, 4, 4, 4, 3, 3, 14, 1, 1, 1, 1];
        let probs = vec64![
            0.45000000000000001,
            0.25,
            0.14999999999999999,
            0.10000000000000001,
            0.050000000000000003
        ];
        let expect = vec64![
            0.0055001488750044125,
            0.00013040370582069444,
            0.00019226894049365975
        ];
        let got = multinomial_pmf(&xs, 18, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_inventory_turnover() {
        let xs = vec64![16u64, 12, 10, 7, 5, 10, 10, 10, 10, 10, 46, 1, 1, 1, 1];
        let probs = vec64![
            0.29999999999999999,
            0.25,
            0.20000000000000001,
            0.14999999999999999,
            0.10000000000000001
        ];
        let expect = vec64![
            0.0006207295903967187,
            1.6072553838585417e-05,
            3.6740423681208188e-21
        ];
        let got = multinomial_pmf(&xs, 50, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn multinomial_pmf_project_phases() {
        let xs = vec64![4u64, 1, 1, 1, 2, 2, 2, 1, 4, 1, 1, 1];
        let probs = vec64![
            0.40000000000000002,
            0.25,
            0.20000000000000001,
            0.14999999999999999
        ];
        let expect = vec64![
            0.040320000000000043,
            0.037800000000000042,
            0.040320000000000043
        ];
        let got = multinomial_pmf(&xs, 7, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_academic_grades() {
        let xs = vec64![14u64, 12, 8, 4, 2, 8, 8, 8, 8, 8, 36, 1, 1, 1, 1];
        let probs = vec64![
            0.34999999999999998,
            0.29999999999999999,
            0.20000000000000001,
            0.10000000000000001,
            0.050000000000000003
        ];
        let expect = vec64![
            0.0014214305417018589,
            1.1312454450349457e-07,
            2.5391056412226081e-14
        ];
        let got = multinomial_pmf(&xs, 40, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn multinomial_pmf_transport_modes() {
        let xs = vec64![12u64, 4, 3, 2, 1, 5, 5, 4, 4, 4, 18, 1, 1, 1, 1];
        let probs = vec64![
            0.5,
            0.20000000000000001,
            0.14999999999999999,
            0.10000000000000001,
            0.050000000000000003
        ];
        let expect = vec64![
            0.0053708244433594452,
            1.7865510428390733e-05,
            0.00010045623779296922
        ];
        let got = multinomial_pmf(&xs, 22, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_quarterly_performance() {
        let xs = vec64![11u64, 10, 7, 7, 9, 9, 9, 8, 32, 1, 1, 1];
        let probs = vec64![
            0.29999999999999999,
            0.29999999999999999,
            0.20000000000000001,
            0.20000000000000001
        ];
        let expect = vec64![
            0.0048130360319179003,
            0.0027234257382251774,
            8.7321723379451621e-15
        ];
        let got = multinomial_pmf(&xs, 35, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn multinomial_pmf_age_demographics() {
        let xs = vec64![
            15u64, 15, 12, 9, 6, 3, 10, 10, 10, 10, 10, 10, 55, 1, 1, 1, 1, 1
        ];
        let probs = vec64![
            0.25,
            0.25,
            0.20000000000000001,
            0.14999999999999999,
            0.10000000000000001,
            0.050000000000000003
        ];
        let expect = vec64![
            0.00011063268076561168,
            1.9112144316469951e-08,
            1.89332811043755e-29
        ];
        let got = multinomial_pmf(&xs, 60, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn multinomial_pmf_multinom_3cat() {
        let xs = vec64![2u64, 3, 5, 1, 4, 5, 3, 3, 4, 0, 5, 5, 5, 5, 0];
        let probs = vec64![
            0.29999999999999999,
            0.40000000000000002,
            0.29999999999999999
        ];
        let expect = vec64![
            0.035271935999999934,
            0.023514624000000057,
            0.05878655999999978,
            0.0062705663999999906,
            0.0062705663999999906
        ];
        let got = multinomial_pmf(&xs, 10, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_multinom_4cat_equal() {
        let xs = vec64![
            2u64, 3, 2, 13, 1, 1, 1, 17, 3, 2, 2, 13, 0, 0, 0, 20, 5, 5, 5, 5
        ];
        let probs = vec64![0.25, 0.25, 0.25, 0.25];
        let expect = vec64![
            1.4805846149101835e-05,
            6.2209437601268399e-09,
            1.4805846149101835e-05,
            9.0949470177292501e-13,
            0.010670869436580696
        ];
        let got = multinomial_pmf(&xs, 20, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn multinomial_pmf_multinom_skewed() {
        let xs = vec64![2u64, 3, 10, 1, 4, 10, 3, 3, 9, 0, 5, 10, 5, 5, 5];
        let probs = vec64![0.5, 0.29999999999999999, 0.20000000000000001];
        let expect = vec64![
            2.0756735999999982e-05,
            6.2270208000000397e-06,
            0.00017297280000000125,
            7.4724249600000414e-07,
            0.018389170800000073
        ];
        let got = multinomial_pmf(&xs, 15, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    // NOTE: The original auto-generated test contained negative counts,
    // which are outside the support of the multinomial and cannot be represented
    // in the kernel’s u64 input. That case has been removed here.

    #[test]
    fn multinomial_pmf_multinom_dominant() {
        let xs = vec64![
            2u64, 3, 2, 1, 1, 1, 1, 5, 3, 2, 2, 1, 0, 0, 0, 8, 2, 2, 2, 2
        ];
        let probs = vec64![
            0.59999999999999998,
            0.20000000000000001,
            0.10000000000000001,
            0.10000000000000001
        ];
        let expect = vec64![
            0.0048384000000000101,
            4.0320000000000054e-05,
            0.014515200000000048,
            9.999999999999982e-09,
            0.0036288000000000045
        ];
        let got = multinomial_pmf(&xs, 8, &probs, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }
}
