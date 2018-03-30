## Development

To build **TensorFlow.js Core API** from source, we need to clone the project and prepare
the dev environment:

```bash
$ git clone https://github.com/tensorflow/tfjs-core.git
$ cd tfjs-core
$ yarn # Installs dependencies.
```

#### Yarn
We use yarn, and if you are adding or removing dependencies you should use yarn
to keep the `yarn.lock` file up to date.

#### Code editor
We recommend using [Visual Studio Code](https://code.visualstudio.com/) for
development. Make sure to install
[TSLint VSCode extension](https://marketplace.visualstudio.com/items?itemName=eg2.tslint)
and the npm [clang-format](https://github.com/angular/clang-format) `1.2.2` or later
with the
[Clang-Format VSCode extension](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format)
for auto-formatting.

#### Testing
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

#### Packaging (browser and npm)

```bash
$ yarn build-npm
> Stored standalone library at dist/tf-core(.min).js
> Stored also tensorflow-tf-core-VERSION.tgz
```

To install it locally, run `yarn add ./tensorflow-tf-core-VERSION.tgz`.

> On Windows, use bash (available through git) to use the scripts above.

Looking to contribute, and don't know where to start? Check out our "help wanted"
[issues](https://github.com/tensorflow/tfjs/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).
