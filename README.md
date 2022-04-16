# spsa

Simultaneous perturbation stochastic approximation implemented in Rust.

- Fast convergence
- Black-box, derivative-free optimization
- Stochastic functions (can be used on noisy functions)
- Converges well in higher-dimensions
- Automatic learning rate tuning and adaptive moment estimation
- Reusable allocation (does not allocate during optimization process)
- SIMD optimization


Based on [this implementation](https://github.com/SimpleArt/spsa) by [SimpleArt](https://github.com/SimpleArt)

[API reference (docs.rs)](https://docs.rs/spsa)


## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
spsa = "0.1.0"
```
# License
`spsa` is distributed under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details.
