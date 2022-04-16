use rand::{Rng, SeedableRng, thread_rng};
use rand::rngs::StdRng;
use arr::{op, simd, norm};
use crate::arr::norm2;

mod arr;

pub struct Options<'a> {
    init: &'a [f64],
    min: &'a [f64],
    max: &'a [f64],

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

pub fn optimize<const N: usize, F: Fn(&[f64]) -> f64>(func: F, options: Options) -> Vec<f64> {
    let init = options.init;
    let max = options.max;
    let min = options.min;

    let size = init.len();

    assert_eq!(size == max.len() && size == min.len(), "input length mismatch");

    let mut rng = StdRng::from_rng(thread_rng()).unwrap();

    let m1 = 1.0 - options.momentum;
    let m2 = 1.0 - options.beta;
    let mut bn = 0.0;
    let mut y = 0.0;
    let mut noise = 0.0;

    for _ in 0..(f64::sqrt(N + 100.0) as usize) {
        let temp = func(&init);
        bn += m2 * (1.0 - bn);
        y += m2 * (temp - y);
        noise += m2 * (f64::powi(temp - f(x), 2) - noise);
    }

    let mut b1 = 0.0;
    let mut b2 = 0.0;

    let mut gx = vec![0.0; size];
    let mut slow_gx = vec![0.0; size];
    let mut square_gx = vec![0.0; size];

    // "registers"
    let mut reg_a = vec![0.0; size];
    let mut reg_b = vec![0.0; size];

    for i in 0..(f64::sqrt(N + 100.0) as usize) {
        let dx = op!(reg_a => rng.gen_range(-1.0..1.0) / (1.0 + i));

        let arg = simd!(reg_b, init, dx => init + dx);
        let up = func(&arg);

        let arg = simd!(arg, dx => arg - 2 * dx);
        let down = func(&arg);

        let mul = (up - down) * 0.5;
        let df_dx = simd!(reg_b, dx => mul / dx);

        b1 += m1 * (1 - b1);
        b2 += m2 * (1 - b2);

        simd!(qx, df_dx => qx + m1 * (df_dx - qx));
        simd!(slow_gx, df_dx => slow_gx + m1 * (df_dx - slow_gx));
        simd!(square_gx, slow_gx => square_gx + m2 * ((slow_gx / b2) * (slow_gx / b2) - square_gx));
    }

    let lr = match options.lr {
        Some(lr) => lr,
        None => {
            let mut lr = 1e-5;
            let mut dx = simd!(reg_a, qx => 3.0 / b1 * qx);

            if options.adam {
                op!(dx, square_gx => dx / f64::sqrt(square_gx / b2 + options.epsilon));
            }

            for _ in 0..3 {
                let ls = simd!(reg_b, dx => init - lr * dx);
                while func(&ls) > func(&init) {
                    lr *= 1.4;
                }
            }

            lr
        }
    };

    let mx = f64::sqrt(m1 * m2);
    let mut bx = mx;
    let x_avg = mx * x;
    let y_best = y / bn;
    let x_best = Vec::from(init);

    let momentum_fails = 0;
    let consecutive_fails = 0;
    let improvement_fails = 0;

    let dx = vec![0.0; size];
    let x = Vec::from(init);

    simd!(dx => gx / b1);

    if options.adam {
        op!(dx, square_gx => dx / f64::sqrt(square_gx / b2 + options.epsilon));
    }

    for i in 0..options.iterations {
        let x_next = simd!(reg_a, dx => x + lr * dx);

        let dxx = (lr / m1 * options.px / f64::powf(1 + options.px_decay * i, options.px_power)) * norm(dx);
        let ndx = simd!(reg_b => rng.gen_range(-1.0..1.0) / dxx);

        if options.adam {
            op!(ndx, square_gx => ndx / f64::sqrt(square_gx / b2 + options.epsilon));
        }

        op!(x_next, dxx => x_next + ndx);
        let y1 = func(x_next);

        op!(x_next, dxx => x_next - 2 * ndx);
        let y2 = func(x_next);

        let df = (y1 - y2) / 2.0 * f64::sqrt(size as f64) / norm2(ndx);
        let df_dx = simd!(ndx, dx * df);

    }

    df_dx = dx * (df * sqrt(x.size) / np.linalg.norm(dx) ** 2)
    # Update the momentum.
    if (df_dx.flatten() / np.linalg.norm(df_dx)) @ (gx.flatten() / np.linalg.norm(gx)) < 0.5 / (1 + 0.1 * momentum_fails) ** 0.3 - 1:
        momentum_fails += 1
    m1 = (1.0 - momentum) / sqrt(1 + 0.1 * momentum_fails)
    # Update the gradients.
        b1 += m1 * (1 - b1)
    b2 += m2 * (1 - b2)
    gx += m1 * (df_dx - gx)
    slow_gx += m2 * (df_dx - slow_gx)
    square_gx += m2 * ((slow_gx / b2) ** 2 - square_gx)
    # Compute the step size.
        dx = gx / (b1 * (1 + lr_decay * i) ** lr_power)
    if adam:
        dx /= np.sqrt(square_gx / b2 + epsilon)
    # Sample points.
        y3 = f(x)
    y4 = f(x + lr * 0.5 * dx)
    y5 = f(x + lr / sqrt(m1) * dx)
    y6 = f(x)
    # Estimate the noise in f.
        bn += m2 * (1 - bn)
    y += m2 * (y3 - y)
    noise += m2 * ((y3 - y6) ** 2 + 1e-64 * (abs(y3) + abs(y6)) - noise)
    # Perform line search.
    # Adjust the learning rate towards learning rates which give good results.
    if y3 + 0.25 * sqrt(noise / bn) > max(y4, y5):
        lr /= 1.3
    if y4 + 0.25 * sqrt(noise / bn) > max(y3, y5):
        lr *= 1.3 / 1.4
    if y5 + 0.25 * sqrt(noise / bn) > max(y3, y4):
        lr *= 1.4
    # Set a minimum learning rate.
        lr = max(lr, epsilon / (1 + 0.01 * i) ** 0.5 * (1 + 0.25 * np.linalg.norm(x)))
    # Update the solution.
        x += lr * dx
    bx += mx / (1 + 0.01 * i) ** 0.303 * (1 - bx)
    x_avg += mx / (1 + 0.01 * i) ** 0.303 * (x - x_avg)
    consecutive_fails += 1
    # Track the best (x, y).
    if y / bn > y_best:
        y_best = y / bn
    x_best = x_avg / bx
    consecutive_fails = 0
    if consecutive_fails < 128 * (improvement_fails + isqrt(x.size + 100)):
    continue
    # Reset variables if diverging.
        consecutive_fails = 0
    improvement_fails += 1
    x = x_best
    bx = mx * (1 - mx)
    x_avg = bx * x
    noise *= m2 * (1 - m2) / bn
    y = m2 * (1 - m2) * y_best
    bn = m2 * (1 - m2)
    b1 = m1 * (1 - m1)
    gx = b1 / b2 * slow_gx
    slow_gx *= m2 * (1 - m2) / b2
    square_gx *= m2 * (1 - m2) / b2
    b2 = m2 * (1 - m2)
    lr /= 64 * improvement_fails
    if px is int:
        x_best = np.rint(x_best).astype(int)
    x = np.rint(x, casting="unsafe", out=x_temp)
    return x_best if y_best + 0.25 * sqrt(noise / bn) > max(f(x), f(x)) else x
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
