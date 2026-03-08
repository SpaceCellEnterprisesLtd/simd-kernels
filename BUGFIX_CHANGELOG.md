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

### M3. Null mask offset bug in (None, Some(m)) pattern
**Files:** `src/kernels/comparison.rs`, `src/kernels/binary.rs`
**Fix:** Split `(Some(m), None) | (None, Some(m))` into separate arms using `lhs_off` and `rhs_off` respectively.
**Rationale:** The OR pattern always used `lhs_off` for both cases, so the RHS-only null mask was sliced at the wrong offset.

### M4. apply_cmp_str/dict ignore offsets when merging masks
**Files:** `src/kernels/binary.rs` (apply_cmp_str, apply_cmp_dict)
**Fix:** Pre-slice masks by offset before passing to `merge_bitmasks_to_new`.
**Rationale:** Full-array masks were merged without windowing, reading bits from wrong positions.

### M5. cmp_dict_between output carries full-array null mask
**File:** `src/kernels/logical.rs:868`
**Fix:** `mask.cloned()` → `mask.map(|m| m.slice_clone(loff, llen))`
**Rationale:** Output carried the full-array mask instead of the windowed slice.

### M6. Scalar erfc() returns 0.0 for NaN
**File:** `src/kernels/scientific/erf.rs:193-195`
**Fix:** Added `if x.is_nan() { return f64::NAN; }` before the infinity branch.
**Rationale:** The `ix >= 0x7ff00000` check matched both NaN and infinity but only returned inf-appropriate values.
**Test needed:** Unit test confirming `erfc(f64::NAN).is_nan()`.

### M7. Binomial PMF NaN at degenerate p=0/p=1
**Files:** `src/kernels/scientific/distributions/univariate/binomial/std.rs`, `binomial/simd.rs`
**Fix:** Added fast paths in scalar_body: `if p == 0.0 { return if ki == 0 { 1.0 } else { 0.0 } }` and same for p=1.0/ki==n.
**Rationale:** `0 * ln(0)` produces NaN. At p=0 only k=0 has probability 1; at p=1 only k=n has probability 1.
**Test needed:** SciPy reference values for p=0 and p=1.

## LOW

### L1. Rolling product integer divide-by-zero
**File:** `src/kernels/window.rs:400-441`
**Fix:** Replaced `rolling_push_pop_to` call with zero-aware push/pop loop tracking `nz_product` and `zero_count` separately.
**Rationale:** When a zero leaves the window, `product / 0` panics for integers. Tracking zeros separately avoids division by zero while keeping O(n) performance.
**Test needed:** Unit test with zeros in the window, e.g. `[1, 0, 3, 4]` window=2.

### L2. rank_float/dense_rank_float NaN panic
**File:** `src/kernels/window.rs:1051,1266`
**Fix:** Replaced `a.partial_cmp(b).unwrap()` with `total_cmp_f` (already exists in `sort.rs`).
**Rationale:** `partial_cmp` returns `None` for NaN, causing `unwrap()` to panic. `total_cmp_f` provides total ordering with NaN > everything.

### L3. softplus overflow for x > ~709
**File:** `src/kernels/scientific/scalar.rs:271-273`
**Fix:** Added fast path: `if x > 20.0 { x }` before `(1.0 + x.exp()).ln()`.
**Rationale:** `e^709` exceeds `f64::MAX`, producing infinity. For x > 20, `softplus(x) ≈ x` to full f64 precision.

### L4. write_global_bitmask_block non-byte-aligned offset
**File:** `src/utils.rs:89-103`
**Fix:** Added bit-shift path for `offset % 8 != 0`: each source byte is split across two destination bytes via `src << bit_off` and `src >> (8 - bit_off)`.
**Rationale:** On non-AVX-512 targets (W64=4 or W64=2), SIMD block offsets are not byte-aligned, causing bits to be written to wrong positions.

## NOT YET APPLIED (remaining from plan)

### L5-L7. Low priority fixes

## DROPPED

### H3. Rolling window first valid result nullified
**Reason:** User review determined the post-processing is intentional design. Tests were written to match this behaviour and the rolling_max reference comparison was inconclusive. Not a bug.
