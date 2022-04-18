mod utils;
mod vec;
mod algo;
mod target;

pub use target::*;

/// Optimization options used in [Optimizer::optimize()]
pub struct Options {
    /// Use adaptive moment estimation
    pub adam: bool,

    /// Maximum number of iterations
    pub iterations: usize,

    /// Learning rate (set to `None` to use estimated value)
    /// The learning rate controls the speed of convergence
    /// ```rust
    /// lr = lr_start / (1 + lr_decay * iteration).pow(lr_power);
    /// x -= lr * gradient_estimate;
    /// ```
    ///
    /// Furthermore, the learning rate is automatically tuned every iteration to produce
    /// improved convergence and allow flexible learning rates.
    pub lr: Option<f64>,

    /// [Learning rate](#structfield.lr) decay
    pub lr_decay: f64,

    /// [Learning rate](#structfield.lr) power
    pub lr_power: f64,

    ///The perturbation size controls how large of a change in x is used to measure changes in f.
    ///This is scaled based on the previous iteration's step size.
    /// ```rust
    /// dx = px / (1.0 + px_decay * i).pow(px_power) * norm(lr * previous_dx) * random_signs;
    /// df = (f(x + dx) - f(x - dx)) / 2.0;
    /// gradient = df / dx;
    /// ```
    pub px: f64,

    /// [Perturbation size](#structfield.px) decay
    pub px_decay: f64,

    /// [Perturbation size](#structfield.px) power
    pub px_power: f64,

    /// The momentum controls how much of the gradient is kept from previous iterations.
    /// Automatically tunes itself to increase as necessary.
    pub momentum: f64,

    /// A secondary momentum, which should be much closer to 1 than the [other momentum](#structfield.momentum).
    /// This is used by the [Adam](#structfield.adam) method.
    pub beta: f64,

    /// Used to avoid division by 0 in the [Adam](#structfield.adam) method.
    pub epsilon: f64
}

impl Default for Options {
    fn default() -> Self {
        Self {
            adam: true,
            iterations: 10_000,
            lr: None,
            lr_decay: 1e-3,
            lr_power: 0.5,
            px: 2.0,
            px_decay: 1e-2,
            px_power: 0.161,
            momentum: 0.9,
            beta: 0.999,
            epsilon: 1e-7
        }
    }
}

/// The heart of this library: a simultaneous perturbation stochastic approximation optimizer
#[derive(Default, Clone)]
pub struct Optimizer([Vec<f64>; algo::REGISTER_NUM]);

/// Optimizer cycle iteration data
pub struct Iteration<'a> {
    pub iteration: usize,
    pub point: &'a mut [f64],
    pub gradient: &'a mut [f64],
    pub learning_rate: &'a mut f64,
}

impl Optimizer {

    /// Allocated memory necessary for optimizer to work
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin SPSA optimizing function `target` starting at `vector`.
    /// Optimized argument vector will be stored in the `vector` argument.
    ///
    /// An optimization process is a process of finding such inputs that maximize an output of some function.
    ///
    /// # Example
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use spsa::{Target, Optimizer, Options};
    ///
    /// struct SimpleFunction;
    ///
    ///  impl Target for SimpleFunction {
    ///     fn evaluate(&mut self, data: &[f64]) -> f64 {
    ///         1.0 - (data[0] + 1.0) * (data[0] + 1.0) - (data[1] - 1.0) * (data[1] - 1.0)
    ///     }
    ///  }
    ///
    ///  let mut optimizer = Optimizer::new();
    ///  let mut input = [0.0, 0.0];
    ///
    ///  optimizer.optimize(SimpleFunction, &mut input, Options::default());
    ///
    ///  assert_relative_eq!(input[0], -1.0, epsilon = 1e-6);
    ///  assert_relative_eq!(input[1],  1.0, epsilon = 1e-6);
    /// ```
    pub fn optimize<T: Target>(&mut self, target: T, vector: &mut [f64], options: Options) {
        let size = vector.len();

        for v in &mut self.0 {
            v.clear();
            v.resize(size, 0.0);
        }

        algo::optimize(target, options, vector, &mut self.0)
    }
}