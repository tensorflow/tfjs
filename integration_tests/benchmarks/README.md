## TensorFlow.js Benchmarks

This directory includes performance benchmarks of TensorFlow.js.
This is work in progress and the command lines and APIs are
subject to frequent changes.

### Running Core Benchmarks

TODO(annyuan): Add doc.

### Running Layers Benchmarks in the Browser

To run the browser-based layers benchmarks, do:

```sh
yarn benchmark --layers
```

Use the `--log` flag to cause the benchmark data and related metadata to be
logged to Cloud Firestore.

```sh
yarn benchmark --layers --log
```

### Running tfjs-node Benchmarks

To run the tfjs-node (CPU) benchmarks, do:

```sh
yarn benchmark --layers --tfjs-node
```

To run the tfjs-node-gpu (CUDA GPU) benchmarks, do:

```sh
yarn benchmark --layers --tfjs-node-gpu
```

Obviously, this requires a CUDA-enabled GPU and all required drivers and
libraries to be set up properly on the system.

Also, note that using the `--tfjs-node-gpu` will cause the the GPU (CUDA)
version of TensorFlow (Python) to be installed and used for comparison.

Add the `--log` flag to cause the benchmark data and related metadata to be
logged to Cloud Firestore. This requires you set the
`GOOGLE_APPLICATION_CREDENTIALS` environment variable and point it
to the service-account JSON file. See this Google Cloud documentation page
for more details:
https://cloud.google.com/docs/authentication/getting-started

