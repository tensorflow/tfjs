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

TODO(cais): Implement and add doc.
