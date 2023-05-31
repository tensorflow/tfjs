# Benchmark on multiple devices

> :warning: To use this tool, you need to sign up for BrowserStack's [Automate](https://automate.browserstack.com/dashboard) service.

The Multi-device benchmark tool can benchmark the performance (time, memory) of model inference on a collection of remote devices. Using this tool you will be able to:
  * Select a collection of BrowserStack devices, based on the following fields:
    - OS
    - OS version
    - Browser
    - Browser version
    - Device
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
2. Download and run the tool:
  ``` shell
  git clone https://github.com/tensorflow/tfjs.git
  cd tfjs/e2e/benchmarks/browserstack-benchmark
  yarn install

  node app.js
  ```
  Then you can see `> Running socket on port: 8001` on your Command-line interface.

3. Open http://localhost:8001/ and start to benchmark.
  3.1 If you want to benchmark code snippet. Please update [`benchmarkCodeSnippet` ](https://github.com/tensorflow/tfjs/pull/6704/files#diff-a7c2ef12f0f2bc1a6cabb45bc9850aa68d10644cd2786e6505456e5537dccadbR92)with your code snippet before running `node app.js` and select `codeSnippet` in `model name`:
<div style="text-align:center">
  <img src="https://user-images.githubusercontent.com/40653845/182249695-021db9a7-e0ef-47e6-8110-ddf777b598a5.png" alt="drawing" height="300px"/>
</div>
4. When the benchmark is complete, you can see the benchmark results in the webpage, like:
<div style="text-align:center">
  <img src="https://user-images.githubusercontent.com/40653845/90341914-a432f180-dfb8-11ea-841e-0d9078c6d50d.png" alt="drawing" height="300px"/>
</div>

### Command Line Arguments
The following are supported options arguments which trigger options features:
  * --benchmarks
    - Runs benchmarks from a user-specified, pre-configured JSON file.
    ``` shell
    node app.js --benchmarks=relative_file_path.json
    ```
    A pre-configuration file consists of a JSON object with the following format:
    ```
    {
      "benchmark": {
        "model": ["model_name"], //List of one or more custom or official models to be benchmarked
        "numRuns": positive_integer,
        "backend": ["backend_name"] //List of one or more backends to be benchmarked
      },
      "browsers": {
        "local": {},  // Benchmark on your local device
        "unique_identifier_laptop_or_desktop": {
          "base": "BrowserStack",
          "browser": "browser_name",
          "browser_version": "browser_version",
          "os": "os_name",
          "os_version": "os_version",
          "device": null
        },
        "unique_identifier_mobile_device": {
          "base": "BrowserStack",
          "browser": "iphone_or_android",
          "browser_version": null,
          "os": "os_name",
          "os_version": "os_version",
          "device": "device_name"
        }
      }
    }
    ```
    Each model in the model list will be run on each backend in the backend list. Each model-backend combination will run on every browser. If you would like to test specific backends on specific models, the recommended method is to create multiple configuration files.

    For more examples of documentation, refer to the links below:
    [List of officially supported TFJS browsers](https://github.com/tensorflow/tfjs/blob/master/e2e/benchmarks/browserstack-benchmark/browser_list.json)
    [Example benchmark pre-configuration](https://github.com/tensorflow/tfjs/blob/master/e2e/benchmarks/browserstack-benchmark/preconfigured_browser.json)
  * --cloud
    - Runs GCP compatible version of benchmarking by blocking the local server.
    ``` shell
    node app.js --cloud
    ```
  * --firestore
    - Pushes successful benchmark results to a Firestore database.
    ``` shell
    node app.js --firestore
    ```
  * --h, --help
    - Shows help menu and all optional arguments in the shell window.
    ``` shell
    node app.js --h
    ```
    or
    ``` shell
    node app.js --help
    ```
  * --period
    - Runs a part of models specified in `--benchmarks`'s file in a cycle and the part of models to run is determined by the date of a month. The value could be or 1~31 (representing Sunday to Saturday). This argument takes effect only if `--benchmarks` is set.
    ``` shell
    node app.js --period=15
    ```
  * --date
    - Set the date for selecting models and this works only if `--period` is set. The value could be 1~31. If it is not declared, the date will the real date at runtime.
    ``` shell
    node app.js --period=15 --date=1
    ```
  * --maxBenchmarks
    - Sets maximum for number of benchmarks run in parallel. Expects a positive integer.
    ``` shell
    node app.js --maxBenchmarks=positive_integer
    ```
  * --maxTries
    - Sets maximum for number of tries a given benchmark has to succeed. Expects a positive integer.
    ``` shell
    node app.js --maxTries=positive_integer
    ```
  * --outfile
    - Writes results to an accessible external file, benchmark_results.js or benchmark_results.json. Expects 'html' or 'json'. If you set it as \'html\', benchmark_results.js will be generated and you could review the benchmark results by openning benchmark_result.html file.
    ``` shell
    node app.js --outfile=js
    ```
  * --v, --version
    - Shows node version in use.
    ``` shell
    node app.js --v
    ```
    or
    ``` shell
    node app.js --version
    ```
  * --localBuild
    - Uses local build dependencies, instead of public CDNs. (**When using localBuild targets, please make sure you have built the targets (eg. run `yarn build-individual-link-package tfjs-backend-webgl`) you need.**)
    ``` shell
    node app.js --localBuild=core,webgl,wasm,cpu,layers,converter,automl
    ```
  * --npmVersion
    - Specify the npm version of TFJS library to benchmark. By default the latest version of TFJS will be benchmarked. 
   ``` shell
    node app.js --npmVersion=4.4.0 
   ```

## Custom model
The custom model is supported, but is constrained by:
  * A URL path to the model is required, while the model in local file system is not supported. The following URLs are examples:
    - TF Hub: https://tfhub.dev/google/tfjs-model/imagenet/resnet_v2_50/feature_vector/1/default/1
    - Storage: https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json
  * Currently only `tf.GraphModel` and `tf.LayersModel` are supported.

If you want to benchmark more complex models with customized input preprocessing logic, you need to add your model with `load` and `predictFunc` methods into [`tfjs/e2e/benchmarks/model_config.js`](https://github.com/tensorflow/tfjs/blob/master/e2e/benchmarks/model_config.js), following this [example PR](https://github.com/tensorflow/tfjs/pull/3168/files).

## About this tool
The tool contains:
  * A test runner - Karma:
    - [benchmark_models.js](https://github.com/tensorflow/tfjs/blob/master/e2e/benchmarks/browserstack-benchmark/benchmark_models.js) warps the all benchmark logics into a Jasmine spec.
    - [browser_list.json](https://github.com/tensorflow/tfjs/blob/master/e2e/benchmarks/browserstack-benchmark/browser_list.json) lists the supported BrowserStack combinations. If you want to add more combinations or refactor this list, you can follow this [conversation](https://github.com/tensorflow/tfjs/pull/3737#issue-463759838).
  * A node server. [app.js](https://github.com/tensorflow/tfjs/blob/master/e2e/benchmarks/browserstack-benchmark/app.js) runs the test runner and send the benchmark results back to the webpage.
  * A webpage.

Thanks, <a href="https://www.browserstack.com/">BrowserStack</a>, for providing supports.
