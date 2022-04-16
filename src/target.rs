use rand::{Rng, SeedableRng, thread_rng};
use rand::rngs::StdRng;
use crate::vec::op;
use crate::utils::rand;

pub trait Target {

    fn evaluate(&mut self, data: &[f64]) -> f64;

    #[allow(unused_variables)]
    fn iteration(&mut self, iter: usize, data: &mut [f64]) {}
}

impl<'a, T: Target> Target for &'a mut T {
    #[inline]
    fn evaluate(&mut self, data: &[f64]) -> f64 {
        (*self).evaluate(data)
    }

    #[inline]
    fn iteration(&mut self, iter: usize, data: &mut [f64]) {
        (*self).iteration(iter, data)
    }
}

pub struct Oversample<T> {
    source: T,
    count: usize
}

impl<T: Target> Target for Oversample<T> {
    fn evaluate(&mut self, data: &[f64]) -> f64 {
        let n = self.count + 1;
        let mut v = 0.0;

        for _ in 0..n {
            v += self.source.evaluate(data);
        }

        v / (n as f64)
    }

    fn iteration(&mut self, iter: usize, data: &mut [f64]) {
        self.source.iteration(iter, data);
    }
}

pub struct OutputNoise<T> {
    source: T,
    gen: StdRng,
    amplitude: f64
}

impl<T: Target> Target for OutputNoise<T> {
    fn evaluate(&mut self, data: &[f64]) -> f64 {
        self.source.evaluate(data) + self.gen.gen_range(-self.amplitude..self.amplitude)
    }

    fn iteration(&mut self, iter: usize, data: &mut [f64]) {
        self.source.iteration(iter, data);
    }
}

pub struct InputNoise<T> {
    source: T,
    gen: StdRng,
    amplitude: f64,
    buffer: Vec<f64>
}

impl<T: Target> Target for InputNoise<T> {
    fn evaluate(&mut self, data: &[f64]) -> f64 {
        let buffer = &mut self.buffer;
        op!(mut buffer => rand(&mut self.gen, -1.0..1.0) * self.amplitude);

        op!(mut buffer, data => buffer + data);
        let up = self.source.evaluate(buffer);

        op!(mut buffer, data => 2.0 * data - buffer);
        let down = self.source.evaluate(buffer);

        (up + down) * 0.5
    }

    fn iteration(&mut self, iter: usize, data: &mut [f64]) {
        self.source.iteration(iter, data);
    }
}

pub trait TargetExt where Self: Sized {

    fn oversample(self, count: usize) -> Oversample<Self> {
        Oversample {
            source: self,
            count
        }
    }

    fn output_noise(self, amplitude: f64) -> OutputNoise<Self> {
        OutputNoise {
            source: self,
            gen: StdRng::from_rng(thread_rng()).unwrap(),
            amplitude
        }
    }

    fn input_noise(self, amplitude: f64) -> InputNoise<Self> {
        InputNoise {
            source: self,
            buffer: Vec::new(),
            gen: StdRng::from_rng(thread_rng()).unwrap(),
            amplitude
        }
    }
}

impl<T: Target> TargetExt for T {}