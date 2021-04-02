# TFLite support for Tensorflow.js

_WORK IN PROGRESS_

This package enables users to run arbitary TFLite models on the web. Users can
load a TFLite model from a URL, use TFJS tensors to set the model's input
data, run inference, and get the output back in TFJS tensors. Under the hood,
the TFLite C++ runtime is packaged in a set of WASM modules, and the one with
the best performance will be automatically loaded based on user's current
environment (e.g. whether WebAssembly SIMD and/or multi-threading is supported
or not).

Check out this [demo][demo] where we use this package to run a
[CartoonGAN][model] TFLite model on the web.


# Development

## Building

```sh
$ yarn
# This script will download the tfweb WASM module files and JS client to deps/.
$ ./script/download-tfweb.sh <version number>
$ yarn build
```

## Testing

```sh
$ yarn test
```

## Deployment
```sh
$ yarn build-npm
# (TODO): publish
```

[demo]: https://storage.googleapis.com/tfweb/demos/cartoonizer/index.html
[model]: https://blog.tensorflow.org/2020/09/how-to-create-cartoonizer-with-tf-lite.html
