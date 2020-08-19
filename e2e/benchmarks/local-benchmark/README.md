# Benchmark custom models

The `custom model` in the [local benchmark tool](https://tensorflow.github.io/tfjs/e2e/benchmarks/local-benchmark/index.html) currently only supports `tf.GraphModel` or `tf.LayersModel`.

If you want to benchmark more complex TensorFlow.js models with customized input preprocessing logic, you need to implement `load` and `predictFunc` methods, following this [example PR](https://github.com/tensorflow/tfjs/pull/3168/files).

## Models in local file system
If you have a TensorFlow.js model in local file system, you can benchmark it through the following steps:
1. Download [tfjs repository](https://github.com/tensorflow/tfjs.git).
2. Under `tfjs/e2e/benchmarks/`, host the model and the tool on a http server:
   1. Make a folder, `model/`.
   2. Put your `model.json` file and the weight files into the `model/` folder.
   3. Run `npx http-server`.
3. Open `http://127.0.0.1:8080/local-benchmark/` through a browser, and you will see the local benchmark tool.
4. Populate the path to the `model.json` file into the `modelUrl` field for the custom model, which is `http://127.0.0.1:8080/model/model.json`.

In addition, if the online [local benchmark tool](https://tensorflow.github.io/tfjs/e2e/benchmarks/local-benchmark/index.html) is blocked by `CORS` problems when fetching custom models, you can locally serve the models by the above steps to solve this problem.

### Example
You can benchmark the [MobileNet model](https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_130_224/classification/3/default/1) in local file system.
1. Download the tool.
```shell
git clone https://github.com/tensorflow/tfjs.git
```
2. Download the model and run a http server.
```shell
cd tfjs/e2e/benchmarks/
wget -O model.tar.gz "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_130_224/classification/3/default/1?tfjs-format=compressed"
mkdir model
tar -xf model.tar.gz -C model/

npx http-server
```
3. Open http://127.0.0.1:8080/local-benchmark/ through the browser.
4. Select `custom` in the `models` field.
5. Fill `http://127.0.0.1:8080/model/model.json` into the `modelUrl` field.
6. Run benchmark.

## Paths to custom models
The benchmark tool suopports three kinds of paths to the custom models.

### URL
Examples:
- TF Hub: https://tfhub.dev/google/tfjs-model/imagenet/resnet_v2_50/feature_vector/1/default/1
- Storage: https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json


### LocalStorage
Store the model in LocalStorage at first. Run the following codes in the browser console:
```javascript
const localStorageModel = tf.sequential(
     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
const saveResults = await localStorageModel.save('localstorage://my-model-1');
```
Then use "localstorage://my-model-1" as the custom model URL.

### IndexDB
Store the model in IndexDB at first. Run the following codes in the browser console:
```javascript
const indexDBModel = tf.sequential(
     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
const saveResults = await indexDBModel.save('indexeddb://my-model-1');
```
Then use "indexeddb://my-model-1" as the custom model URL.
