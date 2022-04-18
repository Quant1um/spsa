use crate::{Iteration, Options, Target};
use crate::utils::{randsign, norm, norm2, cosine, nz};
use crate::vec::op;

use rand::{SeedableRng, thread_rng};
use rand::rngs::StdRng;

pub const REGISTER_NUM: usize = 8;

pub fn optimize<T: Target>(mut target: T, options: Options, x: &mut [f64], r: &mut [Vec<f64>; REGISTER_NUM]) {
    let size = x.len();
    let Options {
        adam,
        iterations,
        lr, lr_decay, lr_power,
        px, px_decay, px_power,
        momentum, beta, epsilon
    } = options;

    let (r0, r) = r.split_first_mut().unwrap();
    let (r1, r) = r.split_first_mut().unwrap();
    let (r2, r) = r.split_first_mut().unwrap();
    let (r3, r) = r.split_first_mut().unwrap();
    let (r4, r) = r.split_first_mut().unwrap();
    let (r5, r) = r.split_first_mut().unwrap();
    let (r6, r) = r.split_first_mut().unwrap();
    let (r7, _) = r.split_first_mut().unwrap();

    let mut rng = StdRng::from_rng(thread_rng()).unwrap();

    let mut m1 = 1.0 - momentum;
    let m2 = 1.0 - beta;
    let mut bn = 0.0;
    let mut y = 0.0;
    let mut noise = 0.0;

    for _ in 0..(f64::sqrt(size as f64 + 100.0) as i32) {
        let temp = target.evaluate(x);
        bn += m2 * (1.0 - bn);
        y += m2 * (temp - y);
        noise += m2 * (f64::powi(temp - target.evaluate(x), 2) - noise);
    }

    if y.is_nan() { // initial point cannot be nan
        x.fill(f64::NAN);
        return;
    }

    let mut b1 = 0.0;
    let mut b2 = 0.0;

    let gx = r0;
    let slow_gx = r1;
    let square_gx = r2;

    for i in 0..(f64::sqrt(size as f64 + 100.0) as i32) {
        let dx = op!(mut r3 => randsign(&mut rng) / (1.0 + i as f64));

        let y1 = target.evaluate(op!(mut r4, x, dx => x + dx));
        let y2 = target.evaluate(op!(mut r4, x, dx => x - dx));

        let df = nz((y1 - y) * 0.5) - nz((y2 - y) * 0.5);
        let df_dx = op!(mut dx => df / dx);

        b1 += m1 * (1.0 - b1);
        b2 += m2 * (1.0 - b2);

        op!(mut gx, df_dx => gx + m1 * (df_dx - gx));
        op!(mut slow_gx, df_dx => slow_gx + m2 * (df_dx - slow_gx));
        op!(mut square_gx, slow_gx => square_gx + m2 * ((slow_gx / b2) * (slow_gx / b2) - square_gx));
    }

    let mut lr = match lr {
        Some(lr) => lr,
        None => {
            let mut lr = 1e-5;
            let dx = op!(mut r3, gx => 3.0 / b1 * gx);

            if adam {
                op!(mut dx, square_gx => dx * (square_gx / b2 + epsilon).rsqrte());
            }

            for _ in 0..5 {
                let ls = op!(mut r4, dx, x => x - lr * dx);

                let a = f64::max(target.evaluate(ls), target.evaluate(ls));
                let b = f64::max(target.evaluate(x), target.evaluate(x));

                if a > b {
                    lr *= 1.4;
                } else {
                    break;
                }
            }

            lr
        }
    };

    let mx = f64::sqrt(m1 * m2);
    let mut bx = mx;

    let x_avg = op!(mut r3, x => mx * x);

    let mut y_best = y / bn;
    let x_best = op!(mut r4, x => x);

    let mut momentum_fails = 0;
    let mut consecutive_fails = 0;
    let mut improvement_fails = 0;

    let dx = op!(mut r5, gx => gx / b1);

    if adam {
        op!(mut dx, square_gx => dx * (square_gx / b2 + epsilon).rsqrte());
    }

    let mut y3 = target.evaluate(x);
    let mut y6 = target.evaluate(x);

    for i in 0..iterations {
        let x_next = op!(mut r6, dx, x => x + lr * dx);

        let dxx = (lr / m1 * px / f64::powf(1.0 + px_decay * i as f64, px_power)) * norm(dx);
        let ndx = op!(mut r7 => randsign(&mut rng) * dxx);

        if adam {
            op!(mut ndx, square_gx => ndx * (square_gx / b2 + epsilon).rsqrte());
        }

        let y1 = target.evaluate(op!(mut x_next, ndx => x_next + ndx));
        let y2 = target.evaluate(op!(mut x_next, ndx => x_next - 2.0 * ndx));
        let df = (nz((y1 - y) * 0.5) - nz((y2 - y) * 0.5)) * f64::sqrt(size as f64) / norm2(ndx);

        if !df.is_finite() {
            break;
        }

        let df_dx = op!(mut ndx => ndx * df);

        if cosine(df_dx, gx) < 0.5 / f64::powf(1.0 + 0.1 * momentum_fails as f64, 0.3) - 1.0 {
            momentum_fails += 1;
            m1 = (1.0 - momentum) / f64::sqrt(1.0 + 0.1 * momentum_fails as f64);
        }

        b1 += m1 * (1.0 - b1);
        b2 += m2 * (1.0 - b2);

        op!(mut gx, df_dx => gx + m1 * (df_dx - gx));
        op!(mut slow_gx, df_dx => slow_gx + m2 * (df_dx - slow_gx));
        op!(mut square_gx, slow_gx => square_gx + m2 * ((slow_gx / b2) * (slow_gx / b2) - square_gx));

        let fa = 1.0 / (b1 * f64::powf(1.0 + lr_decay * i as f64, lr_power));
        op!(mut dx, gx => gx * fa);

        if adam {
            op!(mut dx, square_gx => dx * (square_gx / b2 + epsilon).rsqrte());
        }

        let m1s = f64::sqrt(m1);
        let y4 = target.evaluate(op!(mut r6, x, dx => x + lr * 0.5 * dx));
        let y5 = target.evaluate(op!(mut r6, x, dx => x + lr / m1s * dx));

        bn += m2 * (1.0 - bn);
        y += m2 * (y3 - y);
        noise += m2 * (f64::powi(y3 - y6, 2) + 1e-64 * (y3.abs() + y6.abs()) - noise);

        let noise_factor = f64::sqrt(noise / bn);

        if y3 + 0.25 * noise_factor > f64::max(y4, y5) {
            lr /= 1.3;
        }

        if y4 + 0.25 * noise_factor > f64::max(y3, y5) {
            lr *= 1.3 / 1.4;
        }

        if y5 + 0.25 * noise_factor > f64::max(y3, y4) {
            lr *= 1.4;
        }

        lr = f64::max(lr, epsilon / f64::sqrt(1.0 + 0.01 * i as f64) * (1.0 + 0.25 * norm(&x)));

        let prev = op!(mut r6, x => x);
        op!(mut x, dx => x + dx * lr);

        y3 = target.evaluate(x);
        y6 = target.evaluate(x);

        if !y3.is_finite() || !y6.is_finite() {
            op!(mut x, prev => prev);

            y3 = target.evaluate(x);
            y6 = target.evaluate(x);

            consecutive_fails += 10;

            if !y3.is_finite() || !y6.is_finite() {
                panic!("stuck out of bounds");
            }
        }

        let fa = mx / f64::powf(1.0 + 0.01 * i as f64, 0.303);
        bx += fa * (1.0 - bx);
        op!(mut x_avg, x => x_avg + fa * (x - x_avg));

        consecutive_fails += 1;

        if y / bn > y_best {
            y_best = y / bn;
            op!(mut x_best, x_avg => x_avg / bx);
            consecutive_fails = 0;
        }

        target.iteration(Iteration {
            iteration: i,
            point: x,
            gradient: gx,
            learning_rate: &mut lr
        });

        if consecutive_fails < 128 * (improvement_fails + (f64::sqrt(size as f64 + 100.0) as i32)) {
            continue;
        }

        consecutive_fails = 0;
        improvement_fails += 1;

        x.copy_from_slice(x_best);
        bx = mx * (1.0 - mx);
        op!(mut x_avg, x => x * bx);

        noise *= m2 * (1.0 - m2) / bn;
        y = m2 * (1.0 - m2) * y_best;
        bn = m2 * (1.0 - m2);
        b1 = m1 * (1.0 - m1);

        let fa = m2 * (1.0 - m2) / b2;
        op!(mut gx, slow_gx => b1 / b2 * slow_gx);
        op!(mut slow_gx => slow_gx * fa);
        op!(mut square_gx => square_gx * fa);

        b2 = m2 * (1.0 - m2);
        lr /= 64.0 * improvement_fails as f64;
    }

    if y_best + 0.25 * f64::sqrt(noise / bn) > f64::max(target.evaluate(x), target.evaluate(x)) {
        x.copy_from_slice(x_best);
    }
}