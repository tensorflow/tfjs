# Getting started

**TF Web converter** is an open source library to load a pretrained TensorFlow model into the browser and run inference through deeplearn.js.

## Usage

1. `yarn add @tensorflow/tfjs-converter` or `npm install @tensorflow/tfjs-converter`

2. Use the scripts/converter.py to convert your Tensorflow [SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md). ([details steps](./scripts/README.md))

3. Instantiate the [Model class](./src/executor/model) and run inference. [Example](./demo/mobilenet.ts)

## Development

To build **Tensorflow.js converter** from source, we need to clone the project and prepare
the dev environment:

```bash
$ git clone https://github.com/tensorflow/web.git
$ cd web
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
