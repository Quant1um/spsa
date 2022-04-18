use rand::{Rng, SeedableRng, thread_rng};
use rand::rngs::StdRng;
use crate::Iteration;
use crate::vec::op;
use crate::utils::rand;

/// Represents a function to optimize
pub trait Target {

    /// Evaluate the function at a given point
    /// Return `f64::NAN` if we're out of bounds
    fn evaluate(&mut self, data: &[f64]) -> f64;

    /// Called after every optimizer iteration
    #[allow(unused_variables)]
    fn iteration(&mut self, iter: Iteration) {}
}

impl<'a, T: Target> Target for &'a mut T {
    #[inline]
    fn evaluate(&mut self, data: &[f64]) -> f64 {
        (*self).evaluate(data)
    }

    #[inline]
    fn iteration(&mut self, iter: Iteration) {
        (*self).iteration(iter)
    }
}

/// Decorator that calls underlying function multiple times to smooth out the noise
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

    fn iteration(&mut self, iter: Iteration) {
        self.source.iteration(iter);
    }
}

/// Decorator that adds random noise to function output
pub struct OutputNoise<T> {
    source: T,
    gen: StdRng,
    amplitude: f64
}

impl<T: Target> Target for OutputNoise<T> {
    fn evaluate(&mut self, data: &[f64]) -> f64 {
        self.source.evaluate(data) + self.gen.gen_range(-self.amplitude..self.amplitude)
    }

    fn iteration(&mut self, iter: Iteration) {
        self.source.iteration(iter);
    }
}

/// Some functions have many local minima, causing SPSA and
/// similar methods to run into bad solutions.
///
/// This can, to some extent, be countered using this decorator.
///
/// In this way, SPSA will explore neighboring inputs instead of getting stuck in a "basin".
///
/// If the noise is sufficiently high, and there is a general trend in
/// the direction of the basins towards the best basin,
/// then this will converge to the locally best basin.
pub struct InputNoise<T> {
    source: T,
    gen: StdRng,
    amplitude: f64,
    buffer: Vec<f64>
}

impl<T: Target> Target for InputNoise<T> {
    fn evaluate(&mut self, data: &[f64]) -> f64 {
        let buffer = &mut self.buffer;
        let rng = &mut self.gen;
        let amp = self.amplitude;

        op!(mut buffer => rand(rng, -1.0..1.0) * amp);

        op!(mut buffer, data => buffer + data);
        let up = self.source.evaluate(buffer);

        op!(mut buffer, data => 2.0 * data - buffer);
        let down = self.source.evaluate(buffer);

        (up + down) * 0.5
    }

    fn iteration(&mut self, iter: Iteration) {
        self.source.iteration(iter);
    }
}

/// Extension methods for [`Target`]
pub trait TargetExt where Self: Sized {

    /// Creates a new [`Target`] that "oversamples" this function,
    /// meaning that it will evaluate the function multiple times
    /// and use the average as the result.
    ///
    /// Used for smoothing particularly noisy functions.
    fn oversample(self, count: usize) -> Oversample<Self> {
        Oversample {
            source: self,
            count
        }
    }

    /// Creates a new [`Target`] that adds slight amount of noise to its output.
    fn output_noise(self, amplitude: f64) -> OutputNoise<Self> {
        OutputNoise {
            source: self,
            gen: StdRng::from_rng(thread_rng()).unwrap(),
            amplitude
        }
    }

    /// Some functions have many local minima, causing SPSA and
    /// similar methods to run into bad solutions.
    ///
    /// This can, to some extent, be countered using this decorator.
    ///
    /// In this way, SPSA will explore neighboring inputs instead of getting stuck in a "basin".
    ///
    /// If the noise is sufficiently high, and there is a general trend in
    /// the direction of the basins towards the best basin,
    /// then this will converge to the locally best basin.
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