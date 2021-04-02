# TFLite support for Tensorflow.js

**WORK IN PROGRESS**

# Development

## Building

```sh
$ yarn
# This script will download tfweb WASM module files and JS client to deps/.
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
