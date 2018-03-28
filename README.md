<a id="travis-badge" href="https://travis-ci.org/PAIR-code/deeplearnjs" alt="Build Status">
  <img src="https://travis-ci.org/PAIR-code/deeplearnjs.svg?branch=master" />
</a>

# Getting started

**deeplearn.js** is an open source hardware-accelerated JavaScript library for
machine intelligence. **deeplearn.js** brings performant machine learning
building blocks to the web, allowing you to train neural networks in a browser
or run pre-trained models in inference mode.

We provide an API that closely mirrors the TensorFlow eager API.
**deeplearn.js** was originally developed by the Google Brain PAIR team to build
powerful interactive machine learning tools for the browser, but it can be used
for everything from education, to model understanding, to art projects.

## Usage

You can install this library via yarn/npm: `yarn add deeplearn`
or `npm install deeplearn`

Alternatively you can use a script tag. Here we load it from a CDN.
In this case it will be available as a global variable named `dl`.

You can replace also specify which version to load replacing `@latest` with a specific
version string (e.g. `0.5.0`).

```html
<script src="https://cdn.jsdelivr.net/npm/deeplearn@latest"></script>
<!-- or -->
<script src="https://unpkg.com/deeplearn@latest"></script>
```

#### TypeScript / ES6 JavaScript

Let's add a scalar value to a 1D Tensor. Deeplearn js supports _broadcasting_
the value of scalar over all the elements in the tensor.

```js
import * as dl from 'deeplearn'; // If not loading the script as a global

const a = dl.tensor1d([1, 2, 3]);
const b = dl.scalar(2);

const result = a.add(b); // a is not modified, result is a new tensor
result.data().then(data => console.log(data)); // Float32Array([3, 4, 5]

// Alternatively you can use a blocking call to get the data.
// However this might slow your program down if called repeatedly.
console.log(result.dataSync()); // Float32Array([3, 4, 5]
```

See the [TypeScript starter project](https://github.com/PAIR-code/deeplearnjs/tree/master/starter/typescript/) and the
[ES6 starter project](https://github.com/PAIR-code/deeplearnjs/tree/master/starter/es6/) to get you quickly started.

#### ES5 JavaScript

Let's do the same thing in ES5 Javascript. Remember to include the script tag to
load deeplearn.js

```js
var a = dl.tensor1d([1, 2, 3]);
var b = dl.scalar(2);

var result = a.add(b); // a is not modified, result is a new tensor

// Option 1: With a Promise.
result.data().then(data => console.log(data)); // Float32Array([3, 4, 5])

// Option 2: Synchronous download of data. Blocks the UI.
console.log(result.dataSync());
```

## Development

To build **deeplearn.js** from source, we need to clone the project and prepare
the dev environment:

```bash
$ git clone https://github.com/PAIR-code/deeplearnjs.git
$ cd deeplearnjs
$ yarn # Installs dependencies.
```

#### Yarn vs NPM
Generally speaking it's up to you. Yarn is fully interoperable with npm. You can either do `yarn` or `npm install`. However we use yarn, and if you are adding or removing dependencies you should use yarn to keep the `yarn.lock` file up to date.

#### Code editor
We recommend using [Visual Studio Code](https://code.visualstudio.com/) for
development. Make sure to install
[TSLint VSCode extension](https://marketplace.visualstudio.com/items?itemName=eg2.tslint)
and the npm [clang-format](https://github.com/angular/clang-format) `1.2.2` or later
with the
[Clang-Format VSCode extension](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format)
for auto-formatting.

#### Interactive development
To interactively develop any of the demos (e.g. `demos/nn-art/`):

```bash
$ ./scripts/watch-demo demos/nn-art
>> Starting up http-server, serving ./
>> Available on:
>>   http://127.0.0.1:8080
>> Hit CTRL-C to stop the server
>> 1357589 bytes written to dist/demos/nn-art/bundle.js (0.85 seconds) at 10:34:45 AM
```

Then visit `http://localhost:8080/demos/nn-art/`. The
`watch-demo` script monitors for changes of typescript code and does
incremental compilation (~200-400ms), so users can have a fast edit-refresh
cycle when developing apps.

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
> Stored standalone library at dist/deeplearn(.min).js
> deeplearn-VERSION.tgz
```

To install it locally, run `npm install ./deeplearn-VERSION.tgz`.

> On Windows, use bash (available through git) to use the scripts above.

Looking to contribute, and don't know where to start? Check out our "help wanted"
[issues](https://github.com/PAIR-code/deeplearnjs/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).

## Supported environments

**deeplearn.js** targets environments with WebGL 1.0 or WebGL 2.0. For devices
without the `OES_texture_float` extension, we fall back to fixed precision
floats backed by a `gl.UNSIGNED_BYTE` texture. For platforms without WebGL,
we provide CPU fallbacks.

## Resources

* [Tutorials](https://deeplearnjs.org/docs/tutorials/index.html)
* [API Reference](https://deeplearnjs.org/docs/api/index.html)
* [Demos](https://deeplearnjs.org/index.html#demos)
* [Roadmap](https://deeplearnjs.org/docs/roadmap.html)

## Thanks

<p style="display:flex; align-items:center;">
  <a href="https://www.browserstack.com/">
    <img src="https://www.browserstack.com/images/layout/browserstack-logo-600x315.png" height="70" style="height:70px;">
  </a>
  <span>&nbsp; for providing testing support.</span>
</p>
