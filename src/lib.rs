mod utils;
mod vec;
mod algo;
mod target;

pub use target::*;

pub struct Options {
    adam: bool,
    iterations: usize,

    lr: Option<f64>,
    lr_decay: f64,
    lr_power: f64,

    px: f64,
    px_decay: f64,
    px_power: f64,

    momentum: f64,
    beta: f64,
    epsilon: f64
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

#[derive(Default, Clone)]
pub struct Optimizer([Vec<f64>; algo::REGISTER_NUM]);

impl Optimizer {

    pub fn new() -> Self {
        Self::default()
    }

    pub fn optimize<T: Target>(&mut self, target: T, vector: &mut [f64], options: Options) {
        let size = vector.len();

        for v in &mut self.0 {
            v.clear();
            v.resize(size, 0.0);
        }

        algo::optimize(target, options, vector, &mut self.0)
    }
}