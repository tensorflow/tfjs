# Getting started

**TF Web converter** is an open source library to load a pretrained TensorFlow model into the browser and run inference through JS.

## Usage

`yarn add @tf-web/converter` or `npm install @tf-web/converter`

## Development

To build **TF Web converter** from source, we need to clone the project and prepare
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
$ yarn test --browsers=Chrome --grep='multinomial'
 
> ...
> Chrome 62.0.3202 (Mac OS X 10.12.6): Executed 28 of 1891 (skipped 1863) SUCCESS (6.914 secs / 0.634 secs)
```

To run the tests once and exit the karma process (helpful on Windows):

```bash
$ yarn test --single-run
```
