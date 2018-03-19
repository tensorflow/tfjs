# Getting started

**Tensorflow.js converter** is an open source library to load a pretrained TensorFlow model into the browser and run inference through Tensorflow.js.
It has two main pieces:

1. [Coversion Python script](./scripts/convert.py), converts your Tensorflow SavedModel to web friendly format.
2. [Javascript API](./src/executor/tf_model.ts), simple one line API for inference.

## Inference with converted models

There are three types of files:

* web_model.pb (model)
* weights_manifest.json (weight manifest file)
* group1-shard\*of\* (collection of weight files)

Remember to serve the manifest and weight files with the same url path.

For example, we have the mobilenet models converted and served for you in following location:

```html
  https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/optimized_model.pb
  https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/weights_manifest.json
  https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/group1-shard1of5
  https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/group1-shard2of5
  https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/group1-shard3of5
  https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/group1-shard4of5
  https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/group1-shard5of5
```

1. Install the tfjs-converter npm package

`yarn add @tensorflow/tfjs-converter` or `npm install @tensorflow/tfjs-converter`

2. Instantiate the [TFModel class](./src/executor/tf_model.ts) and run inference. [Example](./demo/mobilenet.ts)


```typescript
import * as tfc from '@tesorflow/tfjs-core';
import {TFModel} from '@tensorflow/tfjs-converter';

const MODEL_FILE_URL = 'http://example.org/models/mobilenet/web_model.pb';
const WEIGHT_MANIFEST_FILE_URL = 'http://example.org/models/mobilenet/weights_manifest.json';

const model = new TFModel(MODEL_FILE_URL, WEIGHT_MANIFEST_FILE_URL);
const cat = document.getElementById('cat');
model.predict({input: tfc.fromPixels(cat)}); // run the inference on your model.
```


## Convert your own Tensorflow pre-trained model in [SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md) format

### Dependencies
1. Clone the github repo:

```bash
  $ git clone git@github.com:tensorflow/tfjs-converter.git
```

2. Install following pip packages:

```bash
  $ pip install tensorflow numpy absl-py protobuf
```

### Conversion

```bash
$ cd tfjs-converter/
$ python scripts/convert.py --saved_model_dir=/tmp/mobilenet/ --output_node_names='MobilenetV1/Predictions/Reshape_1' --output_graph=/tmp/mobilenet/web_model.pb --saved_model_tags=serve
```

| Options         | Description                                                      | Default value |
|---|---|---|
|saved_model_dir  | Full path of the saved model directory                           | |
|output_node_names| The names of the output nodes, comma separated                   | |
|output_graph     | Full path of the name for the output graph file                  | |
|saved_model_tags |SavedModel Tags of the MetaGraphDef to load, in comma separated string format| serve |

### Outputs

This script would generate a collection of files, including model topology file, weight manifest file and weight files.
In the above example, generated files are:

* web_model.pb (model)
* weights_manifest.json (weight manifest file)
* group1-shard\*of\* (collection of weight files)

You can serve these files similarly as shown in the inference [example] (./demo).

### Limitations

Currently Tensorflow.js only supports a limit set of Tensorflow Ops, here is the [full list](./docs/supported_ops.md).
When you converting model with any unsupported Ops, the convert.py script will prompt the unsupported Ops list at the end of the execution. Please fill bugs to let us know what Ops you need support with.


## FAQ

1. What Tensorflow models does the converter currently support?

Image-based models (MobileNet, SqueezeNet, add more if you tested) are the most supported. Models with control flow ops (e.g. RNNs) are not yet supported. The convert.py script will validate the model you have and show a list of unsupported ops in your model. See [this list](./docs/supported_ops.md) for which ops are currently supported.

2. Will model with large weights work?

While the browser supports loading 100-500MB models, the page load time, the inference time and the user experience would not be great. We recommend using models that are designed for edge devices (e.g. phones). These models are usually smaller than 30MB.

3. Will the model and weight files be cached in the browser?

Yes, we are splitting the weights into files of 4MB chunks, which enable the browser to cache them automatically. If the model architecture is less than 4MB (most models are), it will also be cached.

4. Will it support model with quantization?

Not yet. We are planning to add quantization support soon.

5. Why the predict() method for inference is so much slower on the first time then the subsequent calls?

The time of first call also includes the compilation time of WebGL shader programs for the model. After the first call the shader programs are cached, which makes the subsequent calls much faster. You can warm up the cache by calling the predict method with an all zero inputs, right after the completion of the model loading.

## Development

To build **Tensorflow.js converter** from source, we need to clone the project and prepare
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
