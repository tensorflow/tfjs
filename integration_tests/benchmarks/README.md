TensorFlow.js Benchmark Tools
=====

This is a micro benchmark to measure the performance of TensorFlow.js kernel ops.

# Usage 

First install all dependencies. Benchmark tools refers local TensorFlow.js code instead of release versions. 

```
$ yarn 
```

Launch the server to host benchmark application.

```
$ yarn server
```

http://localhost:8080 show the benchmark tool for various kind of kernel ops.

- Batch Normalization 3D
- Matrix multiplication
- Convolutional Ops
- Pooling Ops 
- Unary Ops
- Reduction Ops

Each benchmark suite runs kernel ops with specific size of input in target backend implementation. This benchmark tools support following backends for now.

- CPU
- WebGL

 