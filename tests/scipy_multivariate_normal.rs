//! AUTO-GENERATED: SciPy parity tests for MVN, adapted to current API and Vec64 inputs.
//! Feeds batched points as a flat row-major slice and passes cov as &[&[f64]].

#![allow(clippy::excessive_precision)]

mod util;

#[cfg(feature = "probability_distributions")]
#[cfg(feature = "linear_algebra")]
mod scipy_multivariate_normal_tests {
    use super::util::assert_slice_close;
    use minarrow::{Vec64, vec64};
    use simd_kernels::kernels::scientific::distributions::multivariate::{mvn_logpdf, mvn_pdf};

    #[inline]
    fn flatten_points(xs: &[Vec64<f64>]) -> Vec<f64> {
        if xs.is_empty() {
            return Vec::new();
        }
        let d = xs[0].len();
        let mut out = Vec::with_capacity(xs.len() * d);
        for row in xs {
            debug_assert_eq!(row.len(), d);
            out.extend_from_slice(row.as_slice());
        }
        out
    }

    #[inline]
    fn cov_as_rows<'a>(cov: &'a [Vec64<f64>]) -> Vec<&'a [f64]> {
        cov.iter().map(|r| r.as_slice()).collect()
    }

    #[inline]
    fn run_mvn_pdf(points: &[Vec64<f64>], mean: &Vec64<f64>, cov: &[Vec64<f64>]) -> Vec<f64> {
        let x_flat = flatten_points(points);
        let cov_rows = cov_as_rows(cov);
        mvn_pdf(&x_flat, mean.as_slice(), cov_rows, None, None)
            .unwrap()
            .to_vec()
    }

    #[inline]
    fn run_mvn_logpdf(points: &[Vec64<f64>], mean: &Vec64<f64>, cov: &[Vec64<f64>]) -> Vec<f64> {
        let x_flat = flatten_points(points);
        let cov_rows = cov_as_rows(cov);
        mvn_logpdf(&x_flat, mean.as_slice(), cov_rows, None, None)
            .unwrap()
            .to_vec()
    }

    #[test]
    fn mvn_pdf_standard_2d() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![1.0, 0.0], vec64![0.0, 1.0]];
        let expect = vec64![
            0.15915494309189535,
            0.058549831524319168,
            0.058549831524319168,
            0.12394999430965298,
            0.021539279301848634
        ];
        let got = run_mvn_pdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_pdf_correlated_2d() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![1.0, 0.5], vec64![0.5, 1.0]];
        let expect = vec64![
            0.1837762984739307,
            0.094353897708959245,
            0.094353897708959245,
            0.11146595955293902,
            0.012769411470920382
        ];
        let got = run_mvn_pdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_pdf_negative_corr_pdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![1.0, -0.7], vec64![-0.7, 1.0]];
        let expect = vec64![
            0.22286149708619235,
            0.0079503595644115459,
            0.0079503595644115459,
            0.19238367121310229,
            0.0044148853337747443
        ];
        let got = run_mvn_pdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_logpdf_negative_corr_logpdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![1.0, -0.7], vec64![-0.7, 1.0]];
        let expect = vec64![
            -1.5012047897774625,
            -4.8345381231107947,
            -4.8345381231107947,
            -1.6482636133068742,
            -5.4227734172284414
        ];
        let got = run_mvn_logpdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_pdf_different_vars_pdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![4.0, 0.0], vec64![0.0, 0.25]];
        let expect = vec64![
            0.15915494309189535,
            0.019008347267785906,
            0.019008347267785906,
            0.093562364371238188,
            0.096532352630053928
        ];
        let got = run_mvn_pdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_logpdf_different_vars_logpdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![4.0, 0.0], vec64![0.0, 0.25]];
        let expect = vec64![
            -1.8378770664093453,
            -3.9628770664093453,
            -3.9628770664093453,
            -2.3691270664093453,
            -2.3378770664093453
        ];
        let got = run_mvn_logpdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_pdf_shifted_mean_pdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![2.0, -1.0];
        let cov = vec![vec64![1.0, 0.0], vec64![0.0, 1.0]];
        let expect = vec64![
            0.013064233284684921,
            0.013064233284684921,
            0.0017680517118520169,
            0.045598654639838594,
            0.096532352630053928
        ];
        let got = run_mvn_pdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_logpdf_shifted_mean_logpdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![2.0, -1.0];
        let cov = vec![vec64![1.0, 0.0], vec64![0.0, 1.0]];
        let expect = vec64![
            -4.3378770664093453,
            -4.3378770664093453,
            -6.3378770664093453,
            -3.0878770664093453,
            -2.3378770664093453
        ];
        let got = run_mvn_logpdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_pdf_standard_3d_pdf() {
        let x = vec![
            vec64![0.0, 0.0, 0.0],
            vec64![1.0, 0.0, 0.0],
            vec64![0.0, 1.0, 1.0],
        ];
        let mean = vec64![0.0, 0.0, 0.0];
        let cov = vec![
            vec64![1.0, 0.0, 0.0],
            vec64![0.0, 1.0, 0.0],
            vec64![0.0, 0.0, 1.0],
        ];
        let expect = vec64![
            0.063493635934240983,
            0.038510836890748953,
            0.023358003305431582
        ];
        let got = run_mvn_pdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_logpdf_standard_3d_logpdf() {
        let x = vec![
            vec64![0.0, 0.0, 0.0],
            vec64![1.0, 0.0, 0.0],
            vec64![0.0, 1.0, 1.0],
        ];
        let mean = vec64![0.0, 0.0, 0.0];
        let cov = vec![
            vec64![1.0, 0.0, 0.0],
            vec64![0.0, 1.0, 0.0],
            vec64![0.0, 0.0, 1.0],
        ];
        let expect = vec64![-2.756815599614018, -3.256815599614018, -3.756815599614018];
        let got = run_mvn_logpdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_pdf_weak_corr_pdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![1.0, 0.3], vec64![0.3, 1.0]];
        let expect = vec64![
            0.16683971353257371,
            0.077308412822298667,
            0.077308412822298667,
            0.11673316570227571,
            0.018527041258532558
        ];
        let got = run_mvn_pdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_logpdf_weak_corr_logpdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![1.0, 0.3], vec64![0.3, 1.0]];
        let expect = vec64![
            -1.7907217266737248,
            -2.5599524959044939,
            -2.5599524959044939,
            -2.147864583816582,
            -3.9885239244759219
        ];
        let got = run_mvn_logpdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_pdf_strong_corr_pdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![1.0, 0.9], vec64![0.9, 1.0]];
        let expect = vec64![
            0.36512648068554682,
            0.21570851451891346,
            0.21570851451891346,
            0.029971406664622079,
            9.7931514083285789e-06
        ];
        let got = run_mvn_pdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_logpdf_strong_corr_logpdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![1.0, 0.9], vec64![0.9, 1.0]];
        let expect = vec64![
            -1.0075114629985196,
            -1.5338272524722036,
            -1.5338272524722036,
            -3.5075114629985209,
            -11.533827252472207
        ];
        let got = run_mvn_logpdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_pdf_elliptical_pdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![1.0, 2.0];
        let cov = vec![vec64![2.0, 1.0], vec64![1.0, 3.0]];
        let expect = vec64![
            0.035345081885016533,
            0.05827418831846453,
            0.011765355710462817,
            0.024292295837560866,
            0.015881569030032853
        ];
        let got = run_mvn_pdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_logpdf_elliptical_logpdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![1.0, 2.0];
        let cov = vec![vec64![2.0, 1.0], vec64![1.0, 3.0]];
        let expect = vec64![
            -3.3425960226263953,
            -2.8425960226263953,
            -4.442596022626395,
            -3.7175960226263953,
            -4.1425960226263951
        ];
        let got = run_mvn_logpdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_pdf_concentrated_pdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![0.1, 0.05], vec64![0.05, 0.1]];
        let expect = vec64![
            1.8377629847393067,
            0.0023387992932303774,
            0.0023387992932303774,
            0.012382749588054572,
            4.8206246353985307e-12
        ];
        let got = run_mvn_pdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_logpdf_concentrated_logpdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![0.1, 0.05], vec64![0.05, 0.1]];
        let expect = vec64![
            0.60854906281059051,
            -6.0581176038560756,
            -6.0581176038560756,
            -4.3914509371894086,
            -26.058117603856076
        ];
        let got = run_mvn_logpdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_pdf_wide_spread_pdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![25.0, 0.0], vec64![0.0, 25.0]];
        let expect = vec64![
            0.0063661977236758186,
            0.0061165755404632861,
            0.0061165755404632861,
            0.0063028529979395776,
            0.005876741183054539
        ];
        let got = run_mvn_pdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_logpdf_wide_spread_logpdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![25.0, 0.0], vec64![0.0, 25.0]];
        let expect = vec64![
            -5.0567528912775455,
            -5.0967528912775455,
            -5.0967528912775455,
            -5.0667528912775452,
            -5.1367528912775455
        ];
        let got = run_mvn_logpdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn mvn_pdf_standard_4d_pdf() {
        let x = vec![
            vec64![0.0, 0.0, 0.0, 0.0],
            vec64![1.0, 1.0, 1.0, 1.0],
            vec64![-0.5, -0.5, -0.5, -0.5],
        ];
        let mean = vec64![0.0, 0.0, 0.0, 0.0];
        let cov = vec![
            vec64![1.0, 0.0, 0.0, 0.0],
            vec64![0.0, 1.0, 0.0, 0.0],
            vec64![0.0, 0.0, 1.0, 0.0],
            vec64![0.0, 0.0, 0.0, 1.0],
        ];
        let expect = vec64![
            0.025330295910584451,
            0.0034280827715261588,
            0.015363601089363008
        ];
        let got = run_mvn_pdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_logpdf_standard_4d_logpdf() {
        let x = vec![
            vec64![0.0, 0.0, 0.0, 0.0],
            vec64![1.0, 1.0, 1.0, 1.0],
            vec64![-0.5, -0.5, -0.5, -0.5],
        ];
        let mean = vec64![0.0, 0.0, 0.0, 0.0];
        let cov = vec![
            vec64![1.0, 0.0, 0.0, 0.0],
            vec64![0.0, 1.0, 0.0, 0.0],
            vec64![0.0, 0.0, 1.0, 0.0],
            vec64![0.0, 0.0, 0.0, 1.0],
        ];
        let expect = vec64![
            -3.6757541328186907,
            -5.6757541328186907,
            -4.1757541328186907
        ];
        let got = run_mvn_logpdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_logpdf_extreme_values() {
        let x = vec![vec64![10.0, 10.0], vec64![-10.0, -10.0]];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![1.0, 0.0], vec64![0.0, 1.0]];
        let expect = vec64![-101.83787706640935, -101.83787706640935];
        let got = run_mvn_logpdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn mvn_logpdf_small_variance() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![0.01, 0.0], vec64![0.0, 0.01]];
        let expect = vec64![
            2.7672931195787456,
            -97.232706880421262,
            -97.232706880421262,
            -22.232706880421254,
            -197.23270688042126
        ];
        let got = run_mvn_logpdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_logpdf_large_variance() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![100.0, 0.0], vec64![0.0, 100.0]];
        let expect = vec64![
            -6.4430472523974371,
            -6.4530472523974369,
            -6.4530472523974369,
            -6.4455472523974375,
            -6.4630472523974367
        ];
        let got = run_mvn_logpdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1e-14);
    }

    #[test]
    fn mvn_logpdf_near_singular() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, -1.0],
            vec64![0.5, -0.5],
            vec64![2.0, 0.0],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![1.0, 0.999], vec64![0.999, 1.0]];
        let expect = vec64![
            1.2696770453226192,
            0.76942692026008797,
            0.76942692026008797,
            -248.73032295469099,
            -999.23057307979434
        ];
        let got = run_mvn_logpdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 9.9999999999999998e-13);
    }

    #[test]
    fn mvn_pdf_mvn_2d_correlated() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, 1.0],
            vec64![2.0, -1.0],
            vec64![0.5, 0.5],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![1.0, 0.5], vec64![0.5, 1.0]];
        let expect = vec64![
            0.1837762984739307,
            0.094353897708959245,
            0.024871417406145683,
            0.0017281519181818594,
            0.15556327812622517
        ];
        let got = run_mvn_pdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_logpdf_mvn_2d_logpdf() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, 1.0],
            vec64![2.0, -1.0],
            vec64![0.5, 0.5],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![1.0, 0.5], vec64![0.5, 1.0]];
        let expect = vec64![
            -1.6940360301834549,
            -2.3607026968501215,
            -3.6940360301834549,
            -6.3607026968501224,
            -1.8607026968501217
        ];
        let got = run_mvn_logpdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_pdf_mvn_3d() {
        let x = vec![
            vec64![0.0, 0.0, 0.0],
            vec64![1.0, -1.0, 0.0],
            vec64![2.0, -2.0, 1.0],
            vec64![-1.0, 0.0, -1.0],
        ];
        let mean = vec64![1.0, -1.0, 0.0];
        let cov = vec![
            vec64![2.0, 0.5, 0.0],
            vec64![0.5, 1.0, -0.3],
            vec64![0.0, -0.3, 1.5],
        ];
        let expect = vec64![
            0.012125712049721528,
            0.040606051933671845,
            0.011521368210330478,
            0.0037953398869675129
        ];
        let got = run_mvn_pdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_logpdf_mvn_3d_logpdf() {
        let x = vec![
            vec64![0.0, 0.0, 0.0],
            vec64![1.0, -1.0, 0.0],
            vec64![2.0, -2.0, 1.0],
            vec64![-1.0, 0.0, -1.0],
        ];
        let mean = vec64![1.0, -1.0, 0.0];
        let cov = vec![
            vec64![2.0, 0.5, 0.0],
            vec64![0.5, 1.0, -0.3],
            vec64![0.0, -0.3, 1.5],
        ];
        let expect = vec64![
            -4.4124271181326504,
            -3.203838161077436,
            -4.463551862508929,
            -5.57398131036169
        ];
        let got = run_mvn_logpdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn mvn_pdf_mvn_independent() {
        let x = vec![
            vec64![0.0, 0.0],
            vec64![1.0, 1.0],
            vec64![-1.0, 1.0],
            vec64![2.0, -1.0],
            vec64![0.5, 0.5],
        ];
        let mean = vec64![0.0, 0.0];
        let cov = vec![vec64![2.0, 0.0], vec64![0.0, 3.0]];
        let expect = vec64![
            0.064974733436139701,
            0.04283398421754657,
            0.04283398421754657,
            0.020233341465005501,
            0.058547114800182265
        ];
        let got = run_mvn_pdf(&x, &mean, &cov);
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }
}
