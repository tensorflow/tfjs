---
layout: page
order: 1
---

# Getting started

**deeplearn.js** is an open source hardware-accelerated JavaScript library for
machine intelligence. **deeplearn.js** brings performant machine learning
building blocks to the web, allowing you to train neural networks in a browser
or run pre-trained models in inference mode.

**deeplearn.js** has two APIs, an immediate execution model (think NumPy) and a
deferred execution model mirroring the TensorFlow API.
**deeplearn.js**
was originally developed by the Google Brain PAIR team to build powerful
interactive machine learning tools for the browser, but it can be used for
everything from education, to model understanding, to art projects.

## Usage

#### From JavaScript

Typescript is the preferred language of choice for **deeplearn.js**, however
you can use it with plain JavaScript.

For this use case, you can load the latest version of the library directly from
Google CDN:

```html
<script src="https://storage.googleapis.com/learnjs-data/deeplearn.js"></script>
```

To use a different version, see the
[release](https://github.com/PAIR-code/deeplearnjs/releases) page on GitHub.

#### From TypeScript

To build **deeplearn.js** from source, we need to clone the project and prepare
the dev environment:

```bash
$ git clone https://github.com/PAIR-code/deeplearnjs.git
$ cd deeplearnjs
$ npm run prep # Installs node modules and bower components.
```

To build a standalone library that can be used directly in the browser using a
`<script>` tag:

```bash
$ ./scripts/build-standalone.sh # Builds standalone library.
>> Stored standalone library at dist/deeplearn.js
```

To build a node package/es6 module:

```bash
$ ./scripts/build-npm.sh # Builds npm package.
>> Stored npm package at dist/deeplearn-VERSION.tgz
```

To interactively develop any of the demos (e.g. `demos/nn-art/`):

```bash
$ ./scripts/watch-demo demos/nn-art/nn-art.ts
>> Starting up http-server, serving ./
>> Available on:
>>   http://127.0.0.1:8080
>> Hit CTRL-C to stop the server
>> 1357589 bytes written to dist/demos/nn-art/bundle.js (0.85 seconds) at 10:34:45 AM
```

Then visit `http://localhost:8080/demos/nn-art/nn-art-demo.html`. The
`watch-demo` script monitors for changes of typescript code and does
incremental compilation (~200-400ms), so users can have a fast edit-refresh
cycle when developing apps using **deeplearn.js**.

To run all the tests:

```bash
$ npm run test
```

Before you submit a pull request, make sure the code is clean of lint errors:

```bash
$ npm run lint
```

## Supported environments

**deeplearn.js** targets WebGL 1.0 devices with the `OES_texture_float`
extension and also targets WebGL 2.0 devices. For platforms without WebGL,
we provide CPU fallbacks.

However, currently our demos do not support Mobile, Firefox, and Safari. Please
view them on desktop Chrome for now. We are working to support more devices.
Check back soon!

## Resources

* [Tutorials](docs/tutorials/index.md)
* [API Reference](http://pair-code.github.io/deeplearnjs/docs/api/globals.html)
* [Demos](http://pair-code.github.io/deeplearnjs/index.html#demos)
* [Roadmap](docs/roadmap.md)

This is not an official Google product.
