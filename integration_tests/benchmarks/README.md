TensorFlow.js Benchmark Tools
=====

This is a micro benchmark to measure the performance of TensorFlow.js kernel ops.

# Benchmarks headless usage

Benchmarks will run nightly on browserstack automatically via `yarn test-travis`.
When running benchmarks locally, use:

`yarn benchmark`

This will not write to firebase, it will simply log what would have been written
to firebase.

If you want to push to firebase locally, add --travis to the test script. Be
careful, as this will overwrite other runs of the day.

On travis, during crons, we run:

`yarn benchmark-travis`

This will run the benchmarks with the `--travis` flag flipped and write to
firebase. Note, this only happens on a cron.

# Benchmarks UI usage

While inside the `ui` directory, install all dependencies.

```
$ yarn
```

Launch the server to host benchmark application.

```
$ yarn server
```

http://localhost:8080 shows the benchmark tool for various kind of kernel ops.

- Batch Normalization 3D
- Matrix multiplication
- Convolutional Ops
- Pooling Ops
- Unary Ops
- Reduction Ops

Each benchmark suite runs kernel ops with specific size of input in target backend implementation. This benchmark tools support following backends for now.

- CPU
- WebGL

