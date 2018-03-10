# Getting started

**Tensorflow.js converter** is an open source library to load a pretrained TensorFlow model into the browser and run inference through Tensorflow.js.
It has two main pieces:
1. [Coversion Python script](./scripts/convert.py), converts your Tensorflow SavedModel to web friendly format.
2. [Javascript API](./src/executor/tf_model.ts), simple one line API for inference.

## Dependencies
The python conversion script requires following packages:

```bash
  $ pip install tensorflow numpy absl-py protobuf
```

## Usage

1. `yarn add @tensorflow/tfjs-converter` or `npm install @tensorflow/tfjs-converter`

2. Use the scripts/convert.py to convert your Tensorflow [SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md).

```bash
$ python node_modules/@tensorflow/tfjs-converter/scripts/convert.py --saved_model_dir=/tmp/mobilenet/ --output_node_names='MobilenetV1/Predictions/Reshape_1' --output_graph=/tmp/mobilenet/web_model.pb --saved_model_tags=serve
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

You need to have the model, weight manifest and weight files accessible through url.
And the manifest and weight files should share the the same url path. For example:

```
  http://example.org/models/mobilenet/weights_manifest.json
  http://example.org/models/mobilenet/group1-shard1of2
  http://example.org/models/mobilenet/group1-shard2of2
```

3. Instantiate the [TFModel class](./src/executor/tf_model.ts) and run inference. [Example](./demo/mobilenet.ts)

```typescript
import {TFModel} from 'tfjs-converter';

const MODEL_FILE_URL = 'http://example.org/models/mobilenet/web_model.pb';
const WEIGHT_MANIFEST_FILE_URL = 'http://example.org/models/mobilenet/weights_manifest.json';

const model = new TFModel(MODEL_FILE_URL, WEIGHT_MANIFEST_FILE_URL);
const cat = document.getElementById('cat');
model.predict({input: dl.fromPixels(cat)}) // run the inference on your model.
```

## Development

To build **Tensorflow.js converter** from source, we need to clone the project and prepare
the dev environment:

```bash
$ git clone https://github.com/tensorflow/tfjs-converter.git
$ cd tfjs-converter
$ yarn prep # Installs dependencies.
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
