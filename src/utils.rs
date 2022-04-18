use crate::vec::vectorize;
use rand::Rng;
use rand::rngs::StdRng;
use rand::distributions::uniform::SampleRange;
use packed_simd_2::f64x4;

#[inline]
pub fn norm2(arr: &[f64]) -> f64 {
    vectorize(arr)
        .map(|f| (f * f).sum())
        .sum()
}

#[inline]
pub fn norm(arr: &[f64]) -> f64 {
    norm2(arr).sqrt()
}

#[inline]
pub fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    vectorize(lhs)
        .zip(vectorize(rhs))
        .map(|(l, r)| (l * r).sum())
        .sum()
}

#[inline]
pub fn cosine(lhs: &[f64], rhs: &[f64]) -> f64 {
    let n1 = norm2(lhs);
    let n2 = norm2(rhs);

    if n1 * n2 < 1e-6 {
        return 0.0;
    }

    let dot = dot(lhs, rhs);
    dot / (n1 * n2).sqrt()
}

#[inline]
pub fn rand(rng: &mut StdRng, r: impl SampleRange<f64> + Clone) -> f64x4 {
    f64x4::new(
        rng.gen_range(r.clone()),
        rng.gen_range(r.clone()),
        rng.gen_range(r.clone()),
        rng.gen_range(r)
    )
}

#[inline]
pub fn randsign(rng: &mut StdRng) -> f64x4 {
    fn sign(b: bool) -> f64 {
        if b {
            1.0
        } else {
            -1.0
        }
    }

    f64x4::new(
        sign(rng.gen::<bool>()),
        sign(rng.gen::<bool>()),
        sign(rng.gen::<bool>()),
        sign(rng.gen::<bool>())
    )
}

#[inline]
pub fn nz(f: f64) -> f64 {
    if f.is_nan() {
        0.0
    } else {
        f
    }
}


#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    #[test]
    fn norm2() {
        let a = [1.0, -1.0, 2.0, 1.0];
        let norm = super::norm2(&a);
        assert_relative_eq!(norm, 7.0);
    }

    #[test]
    fn norm() {
        let a = [2.0, -1.0, 2.0, 2.0, 2.0];
        let norm = super::norm(&a);
        assert_relative_eq!(norm, 4.123105625617661);
    }

    #[test]
    fn dot() {
        let a = [2.0, -1.0, 2.0, 2.0, 2.0, -2.0];
        let b = [1.0, 0.0, -2.0, 2.0, 5.0, 1.0];
        let dot = super::dot(&a, &b);
        assert_relative_eq!(dot, 10.0);
    }

    #[test]
    fn cosine() {
        let a = [1.0, 2.0, 3.0];
        let b = [-5.0, -5.0, 5.0];
        let cos = super::cosine(&a, &b);
        assert_relative_eq!(cos, 0.0);
    }
}