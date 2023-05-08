# TensorFlow.js Node.js bindings development.

The @tensorflow/tfjs-node repo supports npm package @tensorflow/tfjs-node and @tensorflow/tfjs-node-gpu on Windows/Mac/Linux. This guide lists commands to use when developing this package.

## Install

#### Dependencies and addon module

```sh
$ yarn
```

This command installs all dependencies and devDependencies listed in package.json. It also downloads the TensorFlow C library and native node addon.

#### Compile native addon from source files

```sh
$ yarn build-addon-from-source
```

This command will compile a new native node addon from source files.

####

```sh
$ yarn install-from-source
```

This command does the following:

1. Clears local binary and addon resources
2. Downloads the TensorFlow C library
3. Compiles the native addon from source files (instead of downloading pre-compile addon)

#### Switching local workflow to CUDA/GPU

```sh
$ yarn enable-gpu
```

This command is the same as `yarn install-from-source` except it uses the TensorFlow GPU library.

## Build and test

#### Compile javascript files from typescript

```sh
$ yarn build
```

#### Publish locally through yalc to test this package in another repo

```sh
$ yarn publish-local
```

This command packs the `tfjs-node` package and publishes locally through [yalc](https://github.com/whitecolor/yalc).
NOTE: Dependent packages must install this locally published package through yalc and compile the node native addon locally. In the dependent package run the following command to link local published `tfjs-node` package:

```sh
$ yalc link @tensorflow/tfjs-node
$ cd .yalc/@tensorflow/tfjs-node
$ yarn && yarn build-addon-from-source
$ cd ../../..
```

#### Run tests

```sh
$ yarn test
```

## Prepare and publish

#### Prerequisite: install GCP command line tool

Publishing this package requires uploading objects to GCP bucket. Developers need to install GCP command line tool [gsutil](https://cloud.google.com/storage/docs/gsutil_install) before publishing. Please ask TFJS developers for GCP project ID.

#### Build and upload node addon to Google Cloud Platform

```sh
$ yarn build-and-upload-addon publish
```

This command will compile, compress, and upload a new node addon to GCP bucket. Please read [build-and-upload-addon.sh](./scripts/build-and-upload-addon.sh) for details.

#### Build NPM package

```sh
$ yarn build-npm
```

This command will build a new version of tfjs-node/tfjs-node-gpu NPM tarball. NOTE: this command does not update the pre-compiled node addon to GCP (see `yarn build-and-upload-addon publish`).

#### Publish NPM package

```sh
$ yarn publish-npm
```

This command compiles a new node addon, upload it to GCP, then builds and publishes a new npm package. Please read instruction in [publish-npm.sh](./scripts/publish.sh) before publishing.

#### Build and upload node addon on Windows

```sh
$ yarn upload-windows-addon
```

Most times the NPM package is published on Linux machine, and only the Linux node addon is compiled and uploaded to GCP bucket. To build and upload the native node addon for Windows, developers should run the above commands on Windows machine. Please read [build-and-upload-windows-addon.bat](./scripts/build-and-upload-windows-addon.bat) for details.

#### Build and upload libtensorflow for custom platforms
Some platforms need a custom version of libtensorflow built for them because tensorflow does not host binaries for them. Right now, the only automated platform is linux-arm64, which can be built with the following command:

```sh
gcloud builds submit . --config=scripts/build-libtensorflow-arm64.yml
```
