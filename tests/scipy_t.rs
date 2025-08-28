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
mod scipy_t_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::student_t::{
        student_t_cdf, student_t_pdf, student_t_quantile,
    };
    // use simd_kernels::kernels::scientific::distributions::univariate::t::{student_t_pdf, student_t_cdf, student_t_quantile};

    #[test]
    fn student_t_pdf_cauchy_equiv() {
        let x = vec64![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let expect = vec64![
            0.031830988618379068,
            0.063661977236758135,
            0.15915494309189535,
            0.31830988618379075,
            0.15915494309189535,
            0.063661977236758135,
            0.031830988618379068
        ];
        let got = student_t_pdf(&x, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn student_t_pdf_heavy_tails() {
        let x = vec64![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let expect = vec64![
            0.027410122234342141,
            0.068041381743977183,
            0.19245008972987526,
            0.35355339059327379,
            0.19245008972987526,
            0.068041381743977183,
            0.027410122234342141
        ];
        let got = student_t_pdf(&x, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn student_t_pdf_moderate_tails() {
        let x = vec64![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let expect = vec64![
            0.01729257880022295,
            0.065090310326216469,
            0.21967979735098056,
            0.3796066898224944,
            0.21967979735098056,
            0.065090310326216469,
            0.01729257880022295
        ];
        let got = student_t_pdf(&x, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn student_t_pdf_lighter_tails() {
        let x = vec64![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let expect = vec64![
            0.011400549464542531,
            0.061145766321218174,
            0.23036198922913867,
            0.38910838396603115,
            0.23036198922913867,
            0.061145766321218174,
            0.011400549464542531
        ];
        let got = student_t_pdf(&x, 10.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn student_t_pdf_approaching_normal() {
        let x = vec64![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let expect = vec64![
            0.006779062746093082,
            0.056852275047197851,
            0.23799334232287925,
            0.39563218489409685,
            0.23799334232287925,
            0.056852275047197851,
            0.006779062746093082
        ];
        let got = student_t_pdf(&x, 30.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn student_t_pdf_large_df() {
        let x = vec64![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let expect = vec64![
            0.0051260897023204053,
            0.054908643295411376,
            0.2407658969285533,
            0.39794618693590594,
            0.2407658969285533,
            0.054908643295411376,
            0.0051260897023204053
        ];
        let got = student_t_pdf(&x, 100.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-13);
    }

    #[test]
    fn student_t_pdf_extreme_values() {
        let x = vec64![-10.0, -5.0, 0.0, 5.0, 10.0];
        let expect = vec64![
            0.00031180821684708739,
            0.0042193537914933053,
            0.36755259694786135,
            0.0042193537914933053,
            0.00031180821684708739
        ];
        let got = student_t_pdf(&x, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 9.9999999999999998e-13);
    }

    #[test]
    fn student_t_cdf_cdf_cauchy() {
        let x = vec64![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let expect = vec64![
            0.10241638234956672,
            0.14758361765043321,
            0.24999999999999978,
            0.5,
            0.75000000000000022,
            0.85241638234956674,
            0.89758361765043326
        ];
        let got = student_t_cdf(&x, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn student_t_cdf_cdf_heavy() {
        let x = vec64![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let expect = vec64![
            0.047732983133354563,
            0.091751709536136955,
            0.21132486540518713,
            0.5,
            0.78867513459481287,
            0.90824829046386302,
            0.9522670168666455
        ];
        let got = student_t_cdf(&x, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn student_t_cdf_cdf_moderate() {
        let x = vec64![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let expect = vec64![
            0.015049623948731284,
            0.05096973941492914,
            0.18160873382456127,
            0.5,
            0.81839126617543867,
            0.9490302605850709,
            0.98495037605126878
        ];
        let got = student_t_cdf(&x, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn student_t_cdf_cdf_lighter() {
        let x = vec64![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let expect = vec64![
            0.0066718275112847827,
            0.036694017385370196,
            0.17044656615103004,
            0.5,
            0.82955343384896996,
            0.96330598261462974,
            0.99332817248871519
        ];
        let got = student_t_cdf(&x, 10.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn student_t_cdf_cdf_near_normal() {
        let x = vec64![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let expect = vec64![
            0.0026949820328259718,
            0.02731252248149155,
            0.16265430771301492,
            0.5,
            0.83734569228698508,
            0.97268747751850848,
            0.99730501796717397
        ];
        let got = student_t_cdf(&x, 30.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn student_t_cdf_cdf_large_df() {
        let x = vec64![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let expect = vec64![
            0.0017039576716647257,
            0.024106089365566821,
            0.1598620778920618,
            0.5,
            0.84013792210793814,
            0.9758939106344332,
            0.99829604232833524
        ];
        let got = student_t_cdf(&x, 100.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-14);
    }

    #[test]
    fn student_t_cdf_cdf_extreme() {
        let x = vec64![-10.0, -5.0, 0.0, 5.0, 10.0];
        let expect = vec64![
            0.0010641995292070747,
            0.0076962190366511481,
            0.5,
            0.99230378096334881,
            0.99893580047079289
        ];
        let got = student_t_cdf(&x, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 9.9999999999999998e-13);
    }

    #[test]
    fn t_ppf_ppf_cauchy() {
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
            -318.3088389855422,
            -31.820515953757607,
            -3.0776835372078062,
            -1.0000000000133888,
            8.1775621612851679e-17,
            1.0000000000133888,
            3.0776835372078066,
            31.820515953757582,
            318.30883898554191
        ];
        let got = student_t_quantile(&q, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-10);
    }

    #[test]
    fn t_ppf_ppf_heavy() {
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
            -10.214531852405337,
            -4.540702858471386,
            -1.6377443536962093,
            -0.76489232840434529,
            7.2037598939616508e-17,
            0.76489232840434529,
            1.6377443536962095,
            4.5407028584713833,
            10.214531852405331
        ];
        let got = student_t_quantile(&q, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-10);
    }

    #[test]
    fn t_ppf_ppf_moderate() {
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
            -4.143700494046624,
            -2.7637694581126961,
            -1.3721836411102861,
            -0.69981206131242912,
            6.8057474240585031e-17,
            0.69981206131242912,
            1.3721836411102863,
            2.7637694581126953,
            4.1437004940466231
        ];
        let got = student_t_quantile(&q, 10.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-13);
    }

    #[test]
    fn t_ppf_ppf_near_normal() {
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
            -3.3851848668182165,
            -2.4572615424005706,
            -1.3104150253913958,
            -0.68275569332129249,
            6.693526796500758e-17,
            0.68275569332129249,
            1.310415025391396,
            2.4572615424005697,
            3.3851848668182161
        ];
        let got = student_t_quantile(&q, 30.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-11);
    }

    #[test]
    fn t_ppf_ppf_extreme() {
        let q = vec64![
            1e-10,
            1.0000000000000001e-05,
            0.5,
            0.99999000000000005,
            0.99999999989999999
        ];
        let expect = vec64![
            -156.82559270889431,
            -15.546854534954758,
            6.976003101422384e-17,
            15.54685453496916,
            156.82559011328064
        ];
        let got = student_t_quantile(&q, 5.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-10);
    }

    #[test]
    fn t_ppf_ppf_large_df() {
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
            -3.1737394937387831,
            -2.3642173662384822,
            -1.2900747613398766,
            -0.67695104300827924,
            6.6546048771515565e-17,
            0.67695104300827924,
            1.2900747613398769,
            2.3642173662384813,
            3.1737394937387822
        ];
        let got = student_t_quantile(&q, 100.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-11);
    }
}
