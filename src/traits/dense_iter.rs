use minarrow::{Bitmask, Vec64};

/// Iterator over non-null values in a slice, using an optional Bitmask.
struct DenseIter<'a, T> {
    slice: &'a [T],
    idx: usize,
    mask: Option<&'a Bitmask>,
    len: usize,
}
impl<'a, T: Copy> DenseIter<'a, T> {
    #[inline(always)]
    fn new(slice: &'a [T], mask: Option<&'a Bitmask>) -> Self {
        let len = slice.len();
        Self {
            slice,
            idx: 0,
            mask,
            len,
        }
    }
}
impl<'a, T: Copy> Iterator for DenseIter<'a, T> {
    type Item = T;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self.mask {
            None => {
                if self.idx >= self.len {
                    None
                } else {
                    let v = self.slice[self.idx];
                    self.idx += 1;
                    Some(v)
                }
            }
            Some(m) => {
                while self.idx < self.len {
                    let i = self.idx;
                    self.idx += 1;
                    if unsafe { m.get_unchecked(i) } {
                        return Some(self.slice[i]);
                    }
                }
                None
            }
        }
    }
}

/// Collects valid (non-null) values from a slice into a Vec64.
#[inline(always)]
pub fn collect_valid<T: Copy>(d: &[T], m: Option<&Bitmask>) -> Vec64<T> {
    DenseIter::new(d, m).collect()
}
