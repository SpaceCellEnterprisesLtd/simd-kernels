//! AUTO-GENERATED: MVN condition-number tests (pdf vs exp(logpdf))
#![allow(clippy::excessive_precision)]

mod util;

#[cfg(feature = "probability_distributions")]
#[cfg(feature = "linear_algebra")]
mod scipy_mvn_condition_tests {
    use super::util::assert_close;
    use simd_kernels::kernels::scientific::distributions::multivariate::{mvn_logpdf, mvn_pdf};

    #[inline]
    fn cov_rows_to_vec<'a, const D: usize>(rows: &'a [[f64; D]; D]) -> Vec<&'a [f64]> {
        rows.iter().map(|r| &r[..]).collect()
    }

    #[test]
    fn mvn_moderate_cond4() {
        // points flattened in row-major blocks of 2
        let x = vec![0.0, 0.0, 1.0, 1.0, 2.0, -1.0, -1.0, 2.0];
        let mean = [0.0, 0.0];
        let cov_rows = [[2.0, 0.5], [0.5, 0.6]];
        let cov = cov_rows_to_vec(&cov_rows);

        // regular pdf path
        let pdf = mvn_pdf(&x, &mean, cov.clone(), None, None).unwrap();
        // stable log path
        let lpdf = mvn_logpdf(&x, &mean, cov, None, None).unwrap();

        assert_eq!(pdf.data.len(), lpdf.data.len());
        for i in 0..pdf.data.len() {
            let e = lpdf.data[i].exp();
            assert_close(pdf.data[i], e, 1e-12);
        }
    }

    #[test]
    fn mvn_moderate_cond10() {
        let x = vec![0.0, 0.0, 1.0, 1.0, 2.0, -1.0, -1.0, 2.0];
        let mean = [0.0, 0.0];
        let cov_rows = [[5.0, 1.5], [1.5, 1.0]];
        let cov = cov_rows_to_vec(&cov_rows);

        let pdf = mvn_pdf(&x, &mean, cov.clone(), None, None).unwrap();
        let lpdf = mvn_logpdf(&x, &mean, cov, None, None).unwrap();

        assert_eq!(pdf.data.len(), lpdf.data.len());
        for i in 0..pdf.data.len() {
            let e = lpdf.data[i].exp();
            assert_close(pdf.data[i], e, 1e-12);
        }
    }

    #[test]
    fn mvn_moderate_cond100() {
        let x = vec![0.0, 0.0, 1.0, 1.0, 2.0, -1.0, -1.0, 2.0];
        let mean = [0.0, 0.0];
        let cov_rows = [[10.0, 0.95], [0.95, 0.1]];
        let cov = cov_rows_to_vec(&cov_rows);

        let pdf = mvn_pdf(&x, &mean, cov.clone(), None, None).unwrap();
        let lpdf = mvn_logpdf(&x, &mean, cov, None, None).unwrap();

        assert_eq!(pdf.data.len(), lpdf.data.len());
        for i in 0..pdf.data.len() {
            let e = lpdf.data[i].exp();
            assert_close(pdf.data[i], e, 5e-11);
        }
    }
}
