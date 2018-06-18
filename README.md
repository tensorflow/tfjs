[![Build Status](https://travis-ci.org/tensorflow/tfjs-converter.svg?branch=master)](https://travis-ci.org/tensorflow/tfjs-converter)

# Getting started

**TensorFlow.js converter** is an open source library to load a pretrained
TensorFlow [SavedModel](https://www.tensorflow.org/programmers_guide/saved_model#overview_of_saving_and_restoring_models), [Frozen Model](https://www.tensorflow.org/mobile/prepare_models#how_do_you_get_a_model_you_can_use_on_mobile) or [Session Bundle](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/session_bundle/README.md)
into the browser and run inference through [TensorFlow.js](https://js.tensorflow.org).

(Note: TensorFlow has deprecated session bundle format, please switch to SavedModel.)

A 2-step process to import your model:

1. A python pip package to convert a TensorFlow SavedModel/Frozen Model/Session Bundle to a web friendly format. If you already have a converted model, or are using an already hosted model (e.g. MobileNet), skip this step.
2. [Javascript API](./src/executor/tf_model.ts), for loading and running inference.

## Step 1: Converting a [SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md), [Keras h5](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model), [Session Bundle](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/session_bundle/README.md), [Frozen Model](https://www.tensorflow.org/mobile/prepare_models#how_do_you_get_a_model_you_can_use_on_mobile) or [Tensorflow Hub module](https://www.tensorflow.org/hub/) to a web-friendly format

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
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    --saved_model_tags=serve \
    /mobilenet/saved_model \
    /mobilenet/web_model
```

Frozen model example:

```bash
$ tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    --saved_model_tags=serve \
    /mobilenet/frozen_model.pb \
    /mobilenet/web_model
```

Session bundle model example:

```bash
$ tensorflowjs_converter \
    --input_format=tf_session_bundle \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    /mobilenet/session_bundle \
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


|Positional Arguments | Description |
|---|---|
|`input_path`  | Full path of the saved model directory, session bundle directory, frozen model file or TensorFlow Hub module handle or path.|
|`output_path` | Path for all output artifacts.|


| Options | Description
|---|---|
|`--input_format`     | The format of input model, use `tf_saved_model` for SavedModel, `tf_frozen_model` for frozen model, `tf_session_bundle` for session bundle, `tf_hub` for TensorFlow Hub module, `tensorflowjs` for TensorFlow.js JSON format, and `keras` for Keras HDF5. |
|<nobr>`--output_node_names`</nobr>| The names of the output nodes, separated by commas.|
|`--output_format`| The desired output format.  Must be `tensorflowjs` (the default) or `keras`.  Not all pairs of input-output formats are supported.  Please file a [github issue](https://github.com/tensorflow/tfjs/issues) if your desired input-output pair is not supported.|
|<nobr>`--saved_model_tags`</nobr> | Only applicable to SavedModel conversion. Tags of the MetaGraphDef to load, in comma separated format. Defaults to `serve`.|
|`--signature_name`   | Only applicable to TensorFlow Hub module conversion, signature to load. Defaults to `default`. See https://www.tensorflow.org/hub/common_signatures/.|


### Format conversions support table

| input format | output `tensorflowjs` | output `keras` |
|---|---|---|
|`keras`| :heavy_check_mark: | :x: |
|`tensorflowjs`| :x: | :heavy_check_mark: |
|`tf_frozen_model`| :heavy_check_mark: | :x: |
|`tf_hub`| :heavy_check_mark: | :x: |
|`tf_saved_model`| :heavy_check_mark: | :x: |
|`tf_session_bundle`| :heavy_check_mark: | :x: |

### Web-friendly format

The conversion script above produces 3 types of files:

* `web_model.pb` (the dataflow graph)
* `weights_manifest.json` (weight manifest file)
* `group1-shard\*of\*` (collection of binary weight files)

For example, here is the MobileNet model converted and served in
following location:

```html
  https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/optimized_model.pb
  https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/weights_manifest.json
  https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/group1-shard1of5
  ...
  https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/group1-shard5of5
```

## Step 2: Loading and running in the browser

1. Install the tfjs-converter npm package

`yarn add @tensorflow/tfjs-converter` or `npm install @tensorflow/tfjs-converter`

2. Instantiate the [FrozenModel class](./src/executor/frozen_model.ts) and run inference.

```typescript
import * as tf from '@tensorflow/tfjs';
import {loadFrozenModel} from '@tensorflow/tfjs-converter';

const MODEL_URL = 'https://.../mobilenet/web_model.pb';
const WEIGHTS_URL = 'https://.../mobilenet/weights_manifest.json';

const model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL);
const cat = document.getElementById('cat');
model.execute({input: tf.fromPixels(cat)});
```

Check out our working [MobileNet demo](./demo/mobilenet/README.md).

If your server requests credentials for accessing the model files, you can provide the optional RequestOption param.

```typescript
const model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL,
    {credentials: 'include'});
```

Please see [fetch() documentation](https://developer.mozilla.org/en-US/docs/Web/API/WindowOrWorkerGlobalScope/fetch) for details.

## Supported operations

Currently TensorFlow.js only supports a limited set of TensorFlow Ops. See the
[full list](./docs/supported_ops.md).
If your model uses an unsupported ops, the `tensorflowjs_converter` script will fail and
produce a list of the unsupported ops in your model. Please file issues to let us
know what ops you need support with.

## Loading the weights only

If you prefer to load the weights only, you can use follow code snippet.

```typescript
import * as tf from '@tensorflow/tfjs';

const weightManifestUrl = "https://example.org/model/weights_manifest.json";

const manifest = await fetch(weightManifestUrl);
this.weightManifest = await manifest.json();
const weightMap = await tf.io.loadWeights(
        this.weightManifest, "https://example.org/model");
```

## FAQ

1. What TensorFlow models does the converter currently support?

Image-based models (MobileNet, SqueezeNet, add more if you tested) are the most supported. Models with control flow ops (e.g. RNNs) are not yet supported. The tensorflowjs_converter script will validate the model you have and show a list of unsupported ops in your model. See [this list](./docs/supported_ops.md) for which ops are currently supported.

2. Will model with large weights work?

While the browser supports loading 100-500MB models, the page load time, the inference time and the user experience would not be great. We recommend using models that are designed for edge devices (e.g. phones). These models are usually smaller than 30MB.

3. Will the model and weight files be cached in the browser?

Yes, we are splitting the weights into files of 4MB chunks, which enable the browser to cache them automatically. If the model architecture is less than 4MB (most models are), it will also be cached.

4. Will it support model with quantization?

Not yet. We are planning to add quantization support soon.

5. Why is the predict() method for inference so much slower on the first call than the subsequent calls?

The time of first call also includes the compilation time of WebGL shader programs for the model. After the first call the shader programs are cached, which makes the subsequent calls much faster. You can warm up the cache by calling the predict method with an all zero inputs, right after the completion of the model loading.

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
[TSLint VSCode extension](https://marketplace.visualstudio.com/items?itemName=eg2.tslint)
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

To generate the static js file for GraphDef proto, run following steps:

1. Generate static js file with comment first, in order to generate typescript definition.

For ES5

```bash
$ node_modules/protobufjs/bin/pbjs -t static-module -w commonjs -o src/data/compiled_api.js --no-create --no-encode --no-verify --no-convert --no-delimited --no-beautify src/data/api.proto

$ node_modules/protobufjs/bin/pbts -o src/data/compiled_api.d.ts src/data/compiled_api.js
```

2. Replace the static js file with the version without comments.
```bash
$ node_modules/protobufjs/bin/pbjs -t static-module -w commonjs -o src/data/compiled_api.js --no-create --no-encode --no-verify --no-convert --no-delimited --no-beautify --no-comments src/data/api.proto
```

For ES6

```bash
$ node_modules/protobufjs/bin/pbjs -t static-module -w es6 -o src/data/es6/compiled_api.js --no-create --no-encode --no-verify --no-convert --no-delimited --no-beautify --no-comments src/data/api.proto

$ sed -i 's/() =>/function()/g' src/data/es6/compiled_api.js
```

Note:
Unfortunately the generated es6 compiled_api.js has arrow function on line 8, the sed command is to replace that with 'function()'
