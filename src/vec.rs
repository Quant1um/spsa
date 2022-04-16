use std::mem::{size_of, transmute};
use std::ptr::copy_nonoverlapping;
use std::ops::{Deref, DerefMut};

//SIMD

pub type Vector = packed_simd_2::f64x4;

#[inline(always)]
fn take_vector(slice: &[f64]) -> Vector {
    if slice.len() > Vector::lanes() {
        let slice = &slice[..Vector::lanes()];
        let mut data = [0.0; Vector::lanes()];
        data.copy_from_slice(slice);
        Vector::from(data)
    } else {
        let mut data = [0.0; Vector::lanes()];
        (&mut data[..slice.len()]).copy_from_slice(slice);
        Vector::from(data)
    }
}

fn prepare_slice<'a, 'b, T>(mut slice: &'a mut [T]) -> &'b mut [T] {
    if slice.len() > Vector::lanes() {
        slice = &mut slice[..Vector::lanes()];
    }

    unsafe { transmute(slice) }
}

#[inline]
pub fn vectorize(data: &[f64]) -> Vectorized {
    Vectorized {
        data,
        index: 0
    }
}

#[inline]
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
    type Item = Vector;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.data.len() {
            return None;
        }

        let slice = &self.data[self.index..];
        self.index += Vector::lanes();

        Some(take_vector(slice))
    }
}

impl<'a> Iterator for VectorizedMut<'a> {
    type Item = MutProxy<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.data.len() {
            return None;
        }

        let slice = &mut self.data[self.index..];
        self.index += Vector::lanes();

        Some(MutProxy {
            data: take_vector(slice),
            slice: prepare_slice(slice)
        })
    }
}

pub struct MutProxy<'a> {
    data: Vector,
    slice: &'a mut [f64]
}

impl<'a> Deref for MutProxy<'a> {
    type Target = Vector;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a> DerefMut for MutProxy<'a> {

    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<'a> Drop for MutProxy<'a> {

    #[inline]
    fn drop(&mut self) {
        unsafe {
            let dst = self.slice.get_unchecked_mut(0) as *mut f64 as *mut u8;
            let src = &self.data as *const Vector as *const u8;
            copy_nonoverlapping(src, dst, (self.slice.len() * size_of::<f64>()).min(size_of::<Vector>()));
        }
    }
}

// OPERATOR

pub trait Operator<T, R> {
    fn run(self, f: impl FnMut(T) -> R);
}

impl<A: AsMut<[f64]>> Operator<Vector, Vector> for A {
    fn run(mut self, mut f: impl FnMut(Vector) -> Vector) {
        for mut a in vectorize_mut(self.as_mut()) {
            *a = f(*a);
        }
    }
}

impl<A: AsMut<[f64]>, B: AsRef<[f64]>> Operator<(Vector, Vector), Vector> for (A, B) {
    fn run(mut self, mut f: impl FnMut((Vector, Vector)) -> Vector) {
        for (mut a, b) in vectorize_mut(self.0.as_mut()).zip(vectorize(self.1.as_ref())) {
            *a = f((*a, b));
        }
    }
}

impl<A: AsMut<[f64]>, B: AsRef<[f64]>, C: AsRef<[f64]>> Operator<(Vector, Vector, Vector), Vector> for (A, B, C) {
    fn run(mut self, mut f: impl FnMut((Vector, Vector, Vector)) -> Vector) {
        for ((mut a, b), c) in vectorize_mut(self.0.as_mut()).zip(vectorize(self.1.as_ref())).zip(vectorize(self.2.as_ref())) {
            *a = f((*a, b, c));
        }
    }
}

macro_rules! op {
    (mut $main:ident $(,$a:ident)* => $e:expr) => {
        {
            #[allow(unused_parens, unused_variables)]
            $crate::vec::Operator::run((&mut *$main $(,&*$a)*), |($main $(,$a)*)| $e);
            &mut *$main
        }
    }
}

pub(crate) use op;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use crate::vec::Operator;

    #[test]
    fn test() {
        let mut a = [10.0, 20.0, 30.0, 40.0, 50.0];
        let b = [20.0, 30.0, 40.0, 50.0, 60.0];

        (&mut a, &b).run(|(a, b)| a + b);

        assert_relative_eq!(a[0], 30.0);
        assert_relative_eq!(a[1], 50.0);
        assert_relative_eq!(a[2], 70.0);
        assert_relative_eq!(a[3], 90.0);
        assert_relative_eq!(a[4], 110.0);
    }
}