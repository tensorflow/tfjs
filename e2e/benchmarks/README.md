# Benchmark custom models

The `custom model` in the [benchmark tool](https://tensorflow.github.io/tfjs/e2e/benchmarks/index.html) currently only supports `tf.GraphModel` or `tf.LayersModel`.

If you want to benchmark models in other types, you need to implement `load` and `predictFunc` methods, following this [example PR](https://github.com/tensorflow/tfjs/pull/3168/files).

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

## Models in local file system
If you have a model in local file system, you can follow the steps below:
1. download [tfjs repository](https://github.com/tensorflow/tfjs.git).
2.  put your `model.json` file and the weight files under the `tfjs/e2e/benchmarks/` folder.
3.  under the `tfjs/e2e/benchmarks/` folder, run `npx http-server`.
4.  open the browser go to `http://127.0.0.1:8080/`, this will open the benchmark tool. Then populate the `model.json` url to `modelUrl` under the custom model, which is `http://127.0.0.1:8080/model.json`.

In addition, if the online tool is blocked by `CORS` problems when fetching the custom model, you can locally serve the model by the above steps to solve this problem.
