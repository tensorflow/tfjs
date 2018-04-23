# TensorFlow backend for TensorFlow.js via Node.js

**This repo is under active development and is not production-ready. We will be
actively developing this in open source. Stay tuned for an official release.**

Currently, we only support developing TensorFlow.js code directly inside of this
repository, however we plan on publishing on NPM soon.

## Trying it out

```sh
# Download and install JS depencies, including libtensorflow 1.8.
yarn

# Publish the NPM locally for usage with other packages.
yarn publish-local
```

See the `demo` directory that trains MNIST using TensorFlow.js with the
TensorFlow C backend.

```sh
cd demo/
yarn

# Link the tfjs-node NPM we published above.
yarn link-local tfjs-node

# Run the training script. See demo/package.json for this script.
yarn mnist
```

The important line to note is at the top of `mnist.ts`, which sets the backend to
TensorFlow.

```js
bindTensorFlowBackend();
```


### Optional: Build libtensorflow From TensorFlow source

This requires installing bazel first.

```sh
bazel build //tensorflow/tools/lib_package:libtensorflow
```

## Supported Platforms

- Mac OS
- Linux
- ***Windows coming soon***
