//! AUTO-GENERATED: discrete PMF sum-to-one tests
#![allow(clippy::excessive_precision)]

mod util;

#[cfg(feature = "probability_distributions")]
mod scipy_sum_to_one_tests {
    use super::util::assert_close;
    use num_traits::Float;

    // Kernel fns
    use simd_kernels::kernels::scientific::distributions::univariate::{
        binomial::binomial_pmf, discrete_uniform::discrete_uniform_pmf,
        hypergeometric::hypergeometric_pmf, poisson::poisson_pmf,
    };

    fn sum(xs: &[f64]) -> f64 {
        xs.iter().copied().sum()
    }

    #[test]
    fn sum_to_one_binomial_n10_p03() {
        let n = 10u64;
        let k: Vec<u64> = (0..=n).collect();
        let out = binomial_pmf(&k, n, 0.3, None, None).unwrap();
        let s = sum(&out);
        assert_close(s, 1.0, 1e-14);
    }

    #[test]
    fn sum_to_one_binomial_n20_p07() {
        let n = 20u64;
        let k: Vec<u64> = (0..=n).collect();
        let out = binomial_pmf(&k, n, 0.7, None, None).unwrap();
        let s = sum(&out);
        assert_close(s, 1.0, 1e-14);
    }

    #[test]
    fn sum_to_one_binomial_n5_p01() {
        let n = 5u64;
        let k: Vec<u64> = (0..=n).collect();
        let out = binomial_pmf(&k, n, 0.1, None, None).unwrap();
        let s = sum(&out);
        assert_close(s, 1.0, 1e-14);
    }

    #[test]
    fn sum_approx_one_poisson_mu_0p5() {
        let mu = 0.5;
        let k_max = (mu + 6.0 * mu.sqrt()).floor() as u64 + 5;
        let k: Vec<u64> = (0..=k_max).collect();
        let out = poisson_pmf(&k, mu, None, None).unwrap();
        let s = sum(&out);
        // allow truncation error (captures >~99.9% of mass)
        assert_close(s, 1.0, 1e-3);
    }

    #[test]
    fn sum_approx_one_poisson_mu_2p0() {
        let mu = 2.0;
        let k_max = (mu + 6.0 * mu.sqrt()).floor() as u64 + 5;
        let k: Vec<u64> = (0..=k_max).collect();
        let out = poisson_pmf(&k, mu, None, None).unwrap();
        let s = sum(&out);
        // allow truncation error (captures >~99.9% of mass)
        assert_close(s, 1.0, 1e-3);
    }

    #[test]
    fn sum_approx_one_poisson_mu_10p0() {
        let mu = 10.0;
        let k_max = (mu + 6.0 * mu.sqrt()).floor() as u64 + 5;
        let k: Vec<u64> = (0..=k_max).collect();
        let out = poisson_pmf(&k, mu, None, None).unwrap();
        let s = sum(&out);
        // allow truncation error (captures >~99.9% of mass)
        assert_close(s, 1.0, 1e-3);
    }

    #[test]
    fn sum_to_one_hypergeom_m20_n7_n12() {
        let m = 20u64;
        let n = 7u64;
        let draws = 12u64;
        let k_min = 0u64.max(draws.saturating_sub(m - n));
        let k_max = n.min(draws);
        let k: Vec<u64> = (k_min..=k_max).collect();
        let out = hypergeometric_pmf(&k, m, n, draws, None, None).unwrap();
        let s = sum(&out);
        assert_close(s, 1.0, 1e-14);
    }

    #[test]
    fn sum_to_one_hypergeom_m10_n5_n5() {
        let m = 10u64;
        let n = 5u64;
        let draws = 5u64;
        let k_min = 0u64.max(draws.saturating_sub(m - n));
        let k_max = n.min(draws);
        let k: Vec<u64> = (k_min..=k_max).collect();
        let out = hypergeometric_pmf(&k, m, n, draws, None, None).unwrap();
        let s = sum(&out);
        assert_close(s, 1.0, 1e-14);
    }

    #[test]
    fn sum_to_one_discrete_uniform_1_7() {
        let low: i64 = 1;
        let high: i64 = 7;
        let k: Vec<i64> = (low..=high).collect();
        let out = discrete_uniform_pmf(&k, low, high, None, None).unwrap();
        let s = sum(&out);
        assert_close(s, 1.0, 1e-14);
    }

    #[test]
    fn sum_to_one_discrete_uniform_0_5() {
        let low: i64 = 0;
        let high: i64 = 5;
        let k: Vec<i64> = (low..=high).collect();
        let out = discrete_uniform_pmf(&k, low, high, None, None).unwrap();
        let s = sum(&out);
        assert_close(s, 1.0, 1e-14);
    }

    #[test]
    fn sum_to_one_discrete_uniform_minus_3_4() {
        let low: i64 = -3;
        let high: i64 = 4;
        let k: Vec<i64> = (low..=high).collect();
        let out = discrete_uniform_pmf(&k, low, high, None, None).unwrap();
        let s = sum(&out);
        assert_close(s, 1.0, 1e-14);
    }
}
