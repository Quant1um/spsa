use approx::assert_relative_eq;
use spsa::{minimize, Optimizer, Options, Target, TargetExt};
use rand::{Rng, SeedableRng, thread_rng};
use rand::rngs::StdRng;

//god its hard to test stochastic functions

#[test]
fn simple_fn() {
    use approx::assert_relative_eq;
    use spsa::{Optimizer, Options, minimize};

    let mut optimizer = Optimizer::new();
    let mut input = [0.0, 0.0];

    optimizer.optimize(minimize(|data| 1.0 + (data[0] + 1.0) * (data[0] + 1.0) + (data[1] - 1.0) * (data[1] - 1.0)), &mut input, Options::default());

    assert_relative_eq!(input[0], -1.0, epsilon = 1e-6);
    assert_relative_eq!(input[1],  1.0, epsilon = 1e-6);
}

#[test]
fn noisy_fn() {
    pub struct NoisyFunction(StdRng);

    impl Target for NoisyFunction {
        fn evaluate(&mut self, data: &[f64]) -> f64 {
            let x = data[0];
            let y = data[1];

            1.0 - (x + 1.0) * (x + 1.0) - (y - 1.0) * (y - 1.0) + self.0.gen_range(-0.001..0.001)
        }
    }

    let mut optimizer = Optimizer::new();
    let mut input = [0.0, 0.0];

    optimizer.optimize(NoisyFunction(StdRng::from_rng(thread_rng()).unwrap()).oversample(2),&mut input, Options::default());

    assert_relative_eq!(input[0], -1.0, epsilon = 1e-2);
    assert_relative_eq!(input[1],  1.0, epsilon = 1e-2);
}

#[test]
fn bounded_fn() {
    pub struct BoundedFunction;

    impl Target for BoundedFunction {
        fn evaluate(&mut self, data: &[f64]) -> f64 {
            let x = data[0];
            let y = data[1];

            let d = 1.0 + (x + 1.0) * (x + 1.0) + (y - 1.0) * (y - 1.0);

            if d > 10.0 || x < 0.0 || y < 0.0 {
                return f64::NAN;
            }

            d
        }
    }

    let mut optimizer = Optimizer::new();
    let mut input = [0.0, 0.0];

    optimizer.optimize(BoundedFunction,&mut input, Options::default());

    assert_relative_eq!(BoundedFunction.evaluate(&input), 10.0, epsilon = 1e-2);
}