use std::mem::size_of;
use std::ops::{Deref, DerefMut, RangeBounds};
use std::ptr::copy_nonoverlapping;
use packed_simd_2::f64x4;
use rand::Rng;
use rand::rngs::StdRng;

pub fn norm2(arr: &[f64]) -> f64 {
    vectorize(arr)
        .map(|f| (f * f).sum())
        .sum()
}

pub fn norm(arr: &[f64]) -> f64 {
    norm2(arr).sqrt()
}

pub fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    vectorize(lhs)
        .zip(vectorize(rhs))
        .map(|(l, r)| (l * r).sum())
        .sum()
}

pub fn cosine(lhs: &[f64], rhs: &[f64]) -> f64 {
    let n1 = norm2(lhs);
    let n2 = norm2(rhs);

    if n1 * n2 < 1e-6 {
        return 0.0;
    }

    let dot = dot(lhs, rhs);
    dot / (n1 * n2).sqrt()
}

pub fn rand(mut rng: StdRng, r: impl RangeBounds<f64> + Copy) -> f64x4 {
    f64x4::new(
        rng.gen_range(r),
        rng.gen_range(r),
        rng.gen_range(r),
        rng.gen_range(r)
    )
}

pub fn vectorize(data: &[f64]) -> Vectorized {
    Vectorized {
        data,
        index: 0
    }
}

pub fn vectorize_mut(data: &mut [f64]) -> VectorizedMut {
    VectorizedMut {
        data,
        index: 0
    }
}

pub struct Vectorized<'a> {
    data: &'a [f64],
    index: usize
}

pub struct VectorizedMut<'a> {
    data: &'a mut [f64],
    index: usize
}

impl<'a> Iterator for Vectorized<'a> {
    type Item = f64x4;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.data.len() {
            return None;
        }

        todo!()
    }
}

impl<'a> Iterator for Vectorized<'a> {
    type Item = MutProxy<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.data.len() {
            return None;
        }

        todo!()
    }
}

pub struct MutProxy<'a> {
    pub data: f64x4,
    slice: &'a mut [f64]
}

impl<'a> Deref for MutProxy<'a> {
    type Target = f64x4;

    #[inline]
    fn deref(&self) -> &f64x4 {
        &self.data
    }
}

impl<'a> DerefMut for MutProxy<'a> {

    #[inline]
    fn deref_mut(&mut self) -> &mut f64x4 {
        &mut self.data
    }
}

impl<'a> Drop for MutProxy<'a> {

    #[inline]
    fn drop(&mut self) {
        unsafe {
            let dest = self.slice.get_unchecked_mut(0) as *mut f64 as *mut u8;
            let src = &self as *const f64x4 as *const u8;
            copy_nonoverlapping(self_ptr, target_ptr, self.slice.len().min(4) * size_of::<f64>());
        }
    }
}