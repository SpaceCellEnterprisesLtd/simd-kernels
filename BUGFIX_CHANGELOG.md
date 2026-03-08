# simd-kernels Bug Fixes Changelog

Tracks all fixes applied, their rationale, and SciPy/unit test requirements.

## CRITICAL

### C1. Wishart PDF normalisation
**File:** `src/kernels/scientific/distributions/multivariate.rs:634`
**Fix:** `df.ln()` → `2.0_f64.ln()`
**Rationale:** Wishart denominator is `2^(nu*d/2)`, requiring `ln(2)` not `ln(nu)`.
**Test needed:** SciPy `scipy.stats.wishart.pdf` reference values for 2x2 and 3x3 matrices.

### C2. Dirichlet PDF normalisation
**File:** `src/kernels/scientific/distributions/multivariate.rs:1160`
**Fix:** `ln_multivariate_gamma(d, sum_alpha)` → `ln_gamma(sum_alpha)`
**Rationale:** Dirichlet uses ordinary gamma, not multivariate gamma. Removed unused `d` variable.
**Test needed:** SciPy `scipy.stats.dirichlet.pdf` reference values for various alpha vectors.

### C3. Wishart sampling matrix multiply
**File:** `src/kernels/scientific/distributions/multivariate.rs:749`
**Fix:** `a[i * d + k]` → `a[k * d + i]`
**Rationale:** LAPACK dpotrf with 'L' stores column-major: `a[col*d + row] = L[row,col]`. The old indexing read L transposed.
**Test needed:** Sample mean convergence test - generate many Wishart samples and verify mean ≈ df * scale.

## HIGH

### H1. not_bool skips negation when null mask present
**File:** `src/kernels/logical.rs:1116-1128`
**Fix:** Removed conditional; always apply NOT to data regardless of null mask.
**Rationale:** The else branch returned `arr.data.slice_clone()` without negation, so `NOT(true)` returned `true` when any null mask was present.
**Test updated:** `unary::tests::not_bool_masked` - changed expected from `0b00001100` to `0b00000011`.

### H2. IsNull/IsNotNull semantics in apply_cmp, apply_cmp_f, apply_cmp_str, apply_cmp_dict
**File:** `src/kernels/binary.rs` (4 functions)
**Fix:** IsNull now returns inverted null mask as data with `null_mask: None`. IsNotNull returns cloned null mask as data with `null_mask: None`. Uses early `return` in str/dict variants to avoid post-match null_mask override.
**Rationale:** Old code returned all-false/all-true data with null_mask preserved, meaning null positions got "unknown" instead of "true"/"false". Reference implementation in `apply_cmp_bool` already did it correctly.
**Test updated:** `binary::tests::test_apply_cmp_numeric_isnull_isnotnull` - position 1 (null) now correctly returns `true` for IsNull, `false` for IsNotNull, with `null_mask: None`.

### H4. Student-t CDF/quantile reject df < 1
**File:** `src/kernels/scientific/distributions/univariate/student_t/std.rs:92,168`
**Fix:** `df < 1.0` → `df <= 0.0` in both CDF and quantile validation.
**Rationale:** t-distribution is mathematically defined for all df > 0. The PDF already used `df <= 0.0`. SciPy accepts any df > 0.
**Test needed:** SciPy reference values for fractional df (e.g. df=0.5) for CDF and quantile.

## MEDIUM

### M1. Variance catastrophic cancellation
**File:** `src/kernels/aggregate.rs:654-674`
**Fix:** Replaced one-pass textbook formula `(s2 - s*s/n)` with two-pass SIMD algorithm:
- Pass 1: SIMD-accelerated compensated sum via existing `stat_moments` → mean
- Pass 2: New SIMD-accelerated `sum_sq_dev` functions computing `sum((x - mean)^2)` with Neumaier compensation
Added new macros: `impl_sum_sq_dev_float`, `impl_sum_sq_dev_int`, `sum_sq_dev!` dispatch.
Instantiated for all 6 types: f64, f32, i64, u64, i32, u32.
**Rationale:** Textbook formula loses precision when mean is large relative to spread. Two-pass preserves SIMD acceleration for both passes.
**Test needed:** Adversarial test with large-offset data, e.g. `[1e12, 1e12+1, 1e12+2]` - variance should be ~1.0, not 0.0.

### M2. cmp_dict_between exclusive lower bound
**File:** `src/kernels/logical.rs:861`
**Fix:** `s > min` → `s >= min`
**Rationale:** Numeric BETWEEN uses `x >= min && x <= max` (inclusive both ends). String BETWEEN same. Dict BETWEEN was inconsistent with exclusive lower bound.
**Test needed:** Unit test with value equal to lower bound confirming it's included.

## NOT YET APPLIED (remaining from plan)

### M3. Null mask offset bug in (None, Some(m)) pattern
### M4. apply_cmp_str/dict ignore offsets when merging masks
### M5. cmp_dict_between output carries full-array null mask
### M6. Scalar erfc() returns 0.0 for NaN
### M7. Binomial PMF NaN at degenerate p=0/p=1
### L1-L7. Low priority fixes

## DROPPED

### H3. Rolling window first valid result nullified
**Reason:** User review determined the post-processing is intentional design. Tests were written to match this behaviour and the rolling_max reference comparison was inconclusive. Not a bug.
