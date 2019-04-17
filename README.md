# Getting started

**TensorFlow.js converter** is an open source library to load a pretrained
TensorFlow [SavedModel](https://www.tensorflow.org/programmers_guide/saved_model#overview_of_saving_and_restoring_models) or [TensorFlow Hub module](https://www.tensorflow.org/hub/)
into the browser and run inference through [TensorFlow.js](https://js.tensorflow.org).

(Note: Session bundle and Frozen model formats have been deprecated in TensorFlow.js 1.0. Please use TensorFlow.js 0.15.x version to convert these formats.)

A 2-step process to import your model:

1. A python pip package to convert a TensorFlow SavedModel or TensorFlow Hub module to a web friendly format. If you already have a converted model, or are using an already hosted model (e.g. MobileNet), skip this step.
2. [Javascript API](./src/executor/tf_model.ts), for loading and running inference.

## Step 1: Converting a [SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md), [Keras h5](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model), [tf.keras SavedModel](https://www.tensorflow.org/api_docs/python/tf/contrib/saved_model/save_keras_model) or [TensorFlow Hub module](https://www.tensorflow.org/hub/) to a web-friendly format

1. Install the TensorFlow.js pip package:

```bash
  $ pip install tensorflowjs
```

2. Run the converter script provided by the pip package:

Usage:

SavedModel example:

```bash
$ tensorflowjs_converter \
    --input_format=tf_saved_model \
    --signature_name=serving_default \
    --saved_model_tags=serve \
    /mobilenet/saved_model \
    /mobilenet/web_model
```

Tensorflow Hub module example:

```bash
$ tensorflowjs_converter \
    --input_format=tf_hub \
    'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1' \
    /mobilenet/web_model
```

Keras h5 model example:

```bash
$ tensorflowjs_converter \
    --input_format=keras \
    /tmp/my_keras_model.h5 \
    /tmp/my_tfjs_model
```

tf.keras SavedModel model example:

```bash
$ tensorflowjs_converter \
    --input_format=keras_saved_model \
    /tmp/my_tf_keras_saved_model/1542211770 \
    /tmp/my_tfjs_model
```

Note that the input path used above is a subfolder that has a Unix epoch
time (1542211770) and is generated automatically by tensorflow when it
saved a tf.keras model in the SavedModel format.

|Positional Arguments | Description |
|---|---|
|`input_path`  | Full path of the saved model directory, session bundle directory, frozen model file or TensorFlow Hub module handle or path.|
|`output_path` | Path for all output artifacts.|


| Options | Description
|---|---|
|`--input_format`     | The format of input model, use `tf_saved_model` for SavedModel, `tf_frozen_model` for frozen model, `tf_session_bundle` for session bundle, `tf_hub` for TensorFlow Hub module, `tfjs_layers_model` for TensorFlow.js JSON format, and `keras` for Keras HDF5. |
|`--output_format`| The desired output format.  Must be `tfjs_layers_model`, `tfjs_graph_model` or `keras`. Not all pairs of input-output formats are supported.  Please file a [github issue](https://github.com/tensorflow/tfjs/issues) if your desired input-output pair is not supported.|
|<nobr>`--saved_model_tags`</nobr> | Only applicable to SavedModel conversion. Tags of the MetaGraphDef to load, in comma separated format. Defaults to `serve`.|
|`--signature_name`   | Only applicable to TensorFlow SavedModel and Hub module conversion, signature to load. Defaults to `serving_default` for SavedModel and `default` for Hub module. See https://www.tensorflow.org/hub/common_signatures/.|
|`--strip_debug_ops`   | Strips out TensorFlow debug operations `Print`, `Assert`, `CheckNumerics`. Defaults to `True`.|
|`--quantization_bytes`  | How many bytes to optionally quantize/compress the weights to. Valid values are 1 and 2. which will quantize int32 and float32 to 1 or 2 bytes respectively. The default (unquantized) size is 4 bytes.|

Note: If you want to convert TensorFlow frozen model or session bundle, you can install older versions of the tensorflowjs pip package, i.e. `pip install tensorflowjs==0.8.0`.

### Format Conversion Support Tables

Note: Unless stated otherwise, we can infer the value of `--output_format` from the
value of `--input_format`. So the `--output_format` flag can be omitted in
most cases.

#### Python-to-JavaScript

| `--input_format` | `--output_format` | Description |
|---|---|---|
| `keras` | `tfjs_layers_model` | Convert a keras or tf.keras HDF5 model file to TensorFlow.js Layers model format. Use [`tf.loadLayersModel()`](https://js.tensorflow.org/api/latest/#loadLayersModel) to load the model in JavaScript. |
| `keras_saved_model` | `tfjs_layers_model` | Convert a tf.keras SavedModel model file (from [`tf.contrib.saved_model.save_keras_model`](https://www.tensorflow.org/api_docs/python/tf/contrib/saved_model/save_keras_model)) to TensorFlow.js Layers model format. Use [`tf.loadLayersModel()`](https://js.tensorflow.org/api/latest/#loadLayersModel) to load the model in JavaScript. |
| `tf_hub` | `tfjs_graph_model` | Convert a [TF-Hub](https://www.tensorflow.org/hub) model file to TensorFlow.js graph model format. Use [`tf.loadGraphModel()`](https://js.tensorflow.org/api/latest/#loadGraphModel) to load the converted model in JavaScript. |
| `tf_saved_model` | `tfjs_graph_model` | Convert a [TensorFlow SavedModel](https://www.tensorflow.org/guide/saved_model#build_and_load_a_savedmodel) to TensorFlow.js graph model format. Use [`tf.loadGraphModel()`](https://js.tensorflow.org/api/latest/#loadGraphModel) to load the converted model in JavaScript. |

#### JavaScript-to-Python

| `--input_format` | `--output_format` | Description |
|---|---|---|
| `tfjs_layers_model` | `keras` | Convert a TensorFlow.js Layers model (JSON + binary weight file(s)) to a Keras HDF5 model file. Use [`keras.model.load_model()`](https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state) or [`tf.keras.models.load_model()`](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model) to load the converted model in Python. |

#### JavaScript-to-JavaScript

The tfjs_layers_model-to-tfjs_layer_model conversion option serves the following
purposes:
1. It allows you to shard the binary weight file into multiple small shards
   to facilitate browser caching. This step is necessary for models with
   large-sized weights saved from TensorFlow.js (either browser or Node.js),
   because TensorFlow.js puts all weights in a single weight file
   ('group1-shard1of1.bin'). To shard the weight file, do

   ```sh
   tensorflowjs_converter \
       --input_format tfjs_layers_model \
       --output_format tfjs_layers_model \
       original_model/model.json \
       sharded_model/
   ```
    The command above creates shards of size 4 MB (4194304 bytes) by default.
    Alternative shard sizes can be specified using the
    `--weight_shard_size_bytes` flag.

2. It allows you to reduce the on-the-wire size of the weights through
   16- or 8-bit quantization. For example:

   ```sh
   tensorflowjs_converter \
      --input_format tfjs_layers_model \
      --output_format tfjs_layers_model \
      --quantization_bytes 2 \
      original_model/model.json
      quantized_model/
   ```

### Web-friendly format

The conversion script above produces 2 types of files:

* `model.json` (the dataflow graph and weight manifest file)
* `group1-shard\*of\*` (collection of binary weight files)

For example, here is the MobileNet model converted and served in
following location:

```html
  https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/model.json
  https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/group1-shard1of5
  ...
  https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/group1-shard5of5
```

## Step 2: Loading and running in the browser

If the original model was a `SavedModel`, use [`tf.loadGraphModel()`](https://js.tensorflow.org/api/latest/#loadGraphModel). If it was Keras, use [`tf.loadLayersModel()`](https://js.tensorflow.org/api/latest/#loadLayersModel):

```typescript
import * as tf from '@tensorflow/tfjs';

const MODEL_URL = 'https://.../mobilenet/model.json';

const model = await tf.loadGraphModel(MODEL_URL); // For Keras use tf.loadLayersModel()
const cat = document.getElementById('cat');
model.predict(tf.browser.fromPixels(cat));
```

See our API docs for the [`predict()`](https://js.tensorflow.org/api/latest/#tf.GraphModel.predict) method. To see what other methods exist on a `Model`, see [`tf.LayersModel`](https://js.tensorflow.org/api/latest/#class:LayersModel) and [`tf.GraphModel`](https://js.tensorflow.org/api/latest/#class:GraphModel). Also check out our working [MobileNet demo](./demo/mobilenet/README.md).

If your server requests credentials for accessing the model files, you can provide the optional RequestOption param.

```typescript
const model = await loadGraphModel(MODEL_URL,
    {credentials: 'include'});
```

Please see [fetch() documentation](https://developer.mozilla.org/en-US/docs/Web/API/WindowOrWorkerGlobalScope/fetch) for details.

### Native File System

TensorFlow.js can be used from Node.js. See
[the tfjs-node project](https://github.com/tensorflow/tfjs-node) for more details.
Unlike web browsers, Node.js can access the local file system directly.
Therefore, you can load the same frozen model from local file system into
a Node.js program running TensorFlow.js. This is done by calling `loadGraphModel` with the path
to the model files:

```js
// Load the tfjs-node binding
import * as tf from '@tensorflow/tfjs-node';

const MODEL_PATH = 'file:///tmp/mobilenet/model.json';
const model = await tf.loadGraphModel(MODEL_PATH);
```

You can also load the remote model files the same way as in browser, but you might need to polyfill
the fetch() method.

## Supported operations

Currently TensorFlow.js only supports a limited set of TensorFlow Ops. See the
[full list](./docs/supported_ops.md).
If your model uses unsupported ops, the `tensorflowjs_converter` script will fail and
produce a list of the unsupported ops in your model. Please file issues to let us
know what ops you need support with.

## Manual forward pass and direct weights loading

If you want to manually write the forward pass with the ops API, you can load the weights directly as a map from weight names to tensors:

```js
import * as tf from '@tensorflow/tfjs';

const modelUrl = "https://example.org/model/model.json";

const response = await fetch(modelUrl);
this.weightManifest = (await response.json())['weightsManifest'];
const weightMap = await tf.io.loadWeights(
        this.weightManifest, "https://example.org/model");
```

`weightMap` maps a weight name to a tensor. You can use it to manually implement the forward pass of the model:

```js
const input = tf.tensor(...);
tf.matMul(weightMap['fc1/weights'], input).add(weightMap['fc1/bias']);
```

## FAQ

1. What TensorFlow models does the converter currently support?

Image-based models (MobileNet, SqueezeNet, add more if you tested) are the most supported. Models with control flow ops (e.g. RNNs) are also supported. The tensorflowjs_converter script will validate the model you have and show a list of unsupported ops in your model. See [this list](./docs/supported_ops.md) for which ops are currently supported.

2. Will model with large weights work?

While the browser supports loading 100-500MB models, the page load time, the inference time and the user experience would not be great. We recommend using models that are designed for edge devices (e.g. phones). These models are usually smaller than 30MB.

3. Will the model and weight files be cached in the browser?

Yes, we are splitting the weights into files of 4MB chunks, which enable the browser to cache them automatically. If the model architecture is less than 4MB (most models are), it will also be cached.

4. Can I quantize the weights over the wire?

Yes, you can use the --quantization_bytes option to compress int32/float32 to 1 or 2 bytes. Here is
an example of 8-bit quantization:

```
tensorflowjs_converter \
    --input_format=tf_hub \
    --quantization_bytes=1
    'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1' \
    /mobilenet/web_model
```

5. Why is the predict() method for inference so much slower on the first call than the subsequent calls?

The time of first call also includes the compilation time of WebGL shader programs for the model. After the first call the shader programs are cached, which makes the subsequent calls much faster. You can warm up the cache by calling the predict method with an all zero inputs, right after the completion of the model loading.

6. I have a model converted with a previous version of TensorFlow.js converter (0.15.x), that is in .pb format. How do I convert it to the new JSON format?

You can use the built-in migration tool to convert the models generated by previous versions. Here are the steps:

```
git clone git@github.com:tensorflow/tfjs-converter.git
cd tfjs-converter
yarn
yarn ts-node tools/pb2json_converter.ts pb_model_directory/ json_model_directory/
```

`pb_model_directory` is the directory where the model generated by previous version is located.
`json_model_directory` is the destination directory for the converted model.

## Development

To build **TensorFlow.js converter** from source, we need to clone the project and prepare
the dev environment:

```bash
$ git clone https://github.com/tensorflow/tfjs-converter.git
$ cd tfjs-converter
$ yarn # Installs dependencies.
```

We recommend using [Visual Studio Code](https://code.visualstudio.com/) for
development. Make sure to install
[TSLint VSCode extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.vscode-typescript-tslint-plugin)
and the npm [clang-format](https://github.com/angular/clang-format) `1.2.2` or later
with the
[Clang-Format VSCode extension](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format)
for auto-formatting.

Before submitting a pull request, make sure the code passes all the tests and is clean of lint errors:

```bash
$ yarn test
$ yarn lint
```

To run a subset of tests and/or on a specific browser:

```bash
$ yarn test --browsers=Chrome --grep='execute'
 
> ...
> Chrome 64.0.3282 (Linux 0.0.0): Executed 39 of 39 SUCCESS (0.129 secs / 0 secs)
```

To run the tests once and exit the karma process (helpful on Windows):

```bash
$ yarn test --single-run
```
