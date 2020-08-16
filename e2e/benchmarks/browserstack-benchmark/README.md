# Benchmark on multiple devices

> :warning: To use this tool, you need to have an access key of BrowserStack's Automate service.

The BrowserStack-benchmark tool can benchmark the performance (time, memory) of model inference on a collection of remote devices. Using this tool you will be able to:
  * Select a collection of BrowserStack devices, based on the following fields:
    - OS
    - OS version
    - browser
    - browser version
    - device
  * Select a backend:
    - WASM
    - WebGL
    - CPU
  * Set the number of rounds for model inference.
  * Select a model to benchmark.

## Usage
1. Export your access key of BrowserStack's Automate service:
  ``` shell
  export BROWSERSTACK_USERNAME=YOUR_USERNAME
  export BROWSERSTACK_ACCESS_KEY=YOUR_ACCESS_KEY
  ```
2. Download and install the tool:
  ``` shell
  git clone https://github.com/Linchenn/tfjs.git
  cd tfjs/e2e/benchmarks/browserstack-benchmark
  yarn install
  ```
  Then you can see `> Running socket on port: 8001` on your Command-line interface.

3. Open http://localhost:8001/ and start to benchmark.

## Custom model
The custom model is supported, but is constrained by:
  * A URL path to the model is required, while the model in local file system is not supported. The following URLs are examples.
    - TF Hub: https://tfhub.dev/google/tfjs-model/imagenet/resnet_v2_50/feature_vector/1/default/1
    - Storage: https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json
  * Currently only `tf.GraphModel` and `tf.LayersModel` are supported.

If you want to benchmark models in other types or customize the inputs for model inference, you need to implement `load` and `predictFunc` methods, following this [example PR](https://github.com/tensorflow/tfjs/pull/3168/files).

## About this tool
The tool mainly contains:
  * A test runner - Karma:
    - [benchmark_models.js](https://github.com/tensorflow/tfjs/blob/master/e2e/benchmarks/browserstack-benchmark/benchmark_models.js) warps the all benchmark logics into a Jasmine spec.
    - [browser_list.json](https://github.com/tensorflow/tfjs/blob/master/e2e/benchmarks/browserstack-benchmark/browser_list.json) lists the supported BrowserStack combinations. If you want to add more combinations or refactor this list, you can follow this [conversation](https://github.com/tensorflow/tfjs/pull/3737#issue-463759838).
  * A node server. [app.js](https://github.com/tensorflow/tfjs/blob/master/e2e/benchmarks/browserstack-benchmark/app.js) runs the test runner and send the benchmark results back to the webpage.
  * A webpage.

Thanks, <a href="https://www.browserstack.com/">BrowserStack</a>, for providing supports.
