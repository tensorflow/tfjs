#!/bin/sh

# TODO(kreeger): Use TF 1.8 builds.
CPU_DARWIN="http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow/TYPE=mac-slave/lastSuccessfulBuild/artifact/lib_package/libtensorflow-cpu-darwin-x86_64.tar.gz"

# Build new package:
# `bazel build //tensorflow/tools/lib_package:libtensorflow`
CPU_LINUX="https://storage.googleapis.com/tf-buiilds/libtensorflow_r1_8.tar.gz"

target=""
platform=$1
if [ "$platform" = "linux-cpu" ]
then
  target=$CPU_LINUX
elif [ "$platform" = "darwin" ]
then
  target=$CPU_DARWIN
else
  echo "Please submit a valid platform"
  exit 1
fi

TARGET_DIRECTORY="deps/tensorflow/"
LIBTENSORFLOW="lib/libtensorflow.so"

mkdir -p $TARGET_DIRECTORY

# Ensure that at least libtensorflow.so is downloaded.
if [ ! -e "$TARGET_DIRECTORY${LIBTENSORFLOW}" ]
then
  curl -L \
    $target |
    tar -C $TARGET_DIRECTORY -xz
fi
