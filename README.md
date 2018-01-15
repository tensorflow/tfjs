<a id="travis-badge" href="https://travis-ci.org/PAIR-code/deeplearnjs" alt="Build Status">
  <img src="https://travis-ci.org/PAIR-code/deeplearnjs.svg?branch=master" />
</a>

# Getting started

**deeplearn.js** is an open source hardware-accelerated JavaScript library for
machine intelligence. **deeplearn.js** brings performant machine learning
building blocks to the web, allowing you to train neural networks in a browser
or run pre-trained models in inference mode.

We provide two APIs, an immediate execution model (think NumPy) and a deferred
execution model mirroring the TensorFlow API.
**deeplearn.js** was originally developed by the Google Brain PAIR team to build
powerful interactive machine learning tools for the browser, but it can be used
for everything from education, to model understanding, to art projects.

## Usage

`yarn add deeplearn` or `npm install deeplearn`

#### TypeScript / ES6 JavaScript
See the [TypeScript starter project](https://github.com/PAIR-code/deeplearnjs/tree/master/starter/typescript/) and the
[ES6 starter project](https://github.com/PAIR-code/deeplearnjs/tree/master/starter/es6/) to get you quickly started. They contain a
short example that sums an array with a scalar (broadcasted):

```ts
import {Array1D, ENV, Scalar} from 'deeplearn';

const math = ENV.math;
const a = Array1D.new([1, 2, 3]);
const b = Scalar.new(2);

const result = math.add(a, b);

// Option 1: With async/await.
// Caveat: in non-Chrome browsers you need to put this in an async function.
console.log(await result.data());  // Float32Array([3, 4, 5])

// Option 2: With a Promise.
result.data().then(data => console.log(data));

// Option 3: Synchronous download of data.
// This is simpler, but blocks the UI until the GPU is done.
console.log(result.dataSync());
```

#### ES3/ES5 JavaScript

You can also use **deeplearn.js** with plain JavaScript. Load the latest version
of the library from [jsDelivr](https://www.jsdelivr.com/) or [unpkg](https://unpkg.com):

```html
<script src="https://cdn.jsdelivr.net/npm/deeplearn"></script>
<!-- or -->
<script src="https://unpkg.com/deeplearn"></script>
```

To use a specific version, add `@version` to the unpkg URL above
(e.g. `https://unpkg.com/deeplearn@0.2.0`), which you can find in the
[releases](https://github.com/PAIR-code/deeplearnjs/releases) page on GitHub.
After importing the library, the API will be available as `dl` in the global
namespace.

```js
var math = dl.ENV.math;
var a = dl.Array1D.new([1, 2, 3]);
var b = dl.Scalar.new(2);

var result = math.add(a, b);

// Option 1: With a Promise.
result.data().then(data => console.log(data)); // Float32Array([3, 4, 5])

// Option 2: Synchronous download of data. This is simpler, but blocks the UI.
console.log(result.dataSync());
```


## Development

To build **deeplearn.js** from source, we need to clone the project and prepare
the dev environment:

```bash
$ git clone https://github.com/PAIR-code/deeplearnjs.git
$ cd deeplearnjs
$ yarn prep # Installs dependencies.
```

#### Yarn vs NPM
It's up to you. Yarn is fully interoperable with npm.  You can either do `yarn` or `npm install`. To add package, you can do `yarn add pgk-name` or `npm install pkg-name`. We use yarn since it is better at caching and resolving dependencies.

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
To build a standalone ES5 library that can be imported in the browser with a
`<script>` tag:

```bash
$ ./scripts/build-standalone.sh # Builds standalone library.
>> Stored standalone library at dist/deeplearn(.min).js
```

To build an npm package:

```bash
$ ./scripts/build-npm.sh
...
Stored standalone library at dist/deeplearn(.min).js
deeplearn-VERSION.tgz
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

While the library supports most devices, our demos don't currently work on
iOS Mobile or Desktop Safari. We are working on updating them, check back soon.

## Resources

* [Tutorials](https://deeplearnjs.org/docs/tutorials/index.html)
* [API Reference](https://deeplearnjs.org/docs/api/globals.html)
* [Demos](https://deeplearnjs.org/index.html#demos)
* [Roadmap](https://deeplearnjs.org/docs/roadmap.html)

## Thanks

<p style="display:flex; align-items:center;">
  <a href="https://www.browserstack.com/">
    <img src="https://www.browserstack.com/images/layout/browserstack-logo-600x315.png" height="70" style="height:70px;">
  </a>
  <span>&nbsp; for providing testing support.</span>
</p>
