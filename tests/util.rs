#![allow(unused)]

pub fn assert_close(a: f64, e: f64, tol: f64) {
    if e.is_nan() {
        assert!(a.is_nan(), "expected NaN, got {a}");
        return;
    }
    if e.is_infinite() {
        assert!(
            a.is_infinite() && a.is_sign_positive() == e.is_sign_positive(),
            "expected {e}, got {a}"
        );
        return;
    }
    let scale = 1.0_f64.max(e.abs());
    let ok = (a - e).abs() <= tol * scale;
    assert!(ok, "mismatch: got {a}, expect {e} (tol={tol})");
}

pub fn assert_slice_close(a: &[f64], e: &[f64], tol: f64) {
    assert_eq!(a.len(), e.len(), "len mismatch");
    for (i, (&ai, &ei)) in a.iter().zip(e.iter()).enumerate() {
        if ei.is_nan() {
            assert!(ai.is_nan(), "idx {i}: expected NaN, got {ai}");
            continue;
        }
        if ei.is_infinite() {
            assert!(
                ai.is_infinite() && ai.is_sign_positive() == ei.is_sign_positive(),
                "idx {i}: expected {ei}, got {ai}"
            );
            continue;
        }
        let scale = 1.0_f64.max(ei.abs());
        let ok = (ai - ei).abs() <= tol * scale;
        assert!(ok, "idx {i}: got {ai}, expect {ei} (tol={tol})");
    }
}
