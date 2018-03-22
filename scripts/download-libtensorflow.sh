#!/bin/sh

CPU_DARWIN="http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow/TYPE=mac-slave/lastSuccessfulBuild/artifact/lib_package/libtensorflow-cpu-darwin-x86_64.tar.gz"
GPU_LINUX="http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow/TYPE=gpu-linux/lastSuccessfulBuild/artifact/lib_package/libtensorflow-gpu-linux-x86_64.tar.gz"
# TODO(kreeger): Use nightly links when Jenkins TF uses an updated bazel.
# Build new package:
# `bazel build --config opt //tensorflow/tools/lib_package:libtensorflow`
CPU_LINUX="https://storage.googleapis.com/tf-buiilds/libtensorflow.tar.gz"
# CPU_LINUX="http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow/TYPE=cpu-slave/lastSuccessfulBuild/artifact/lib_package/libtensorflow-cpu-linux-x86_64.tar.gz"

target=""
platform=$1
if [ "$platform" = "linux-cpu" ]
then
  target=$CPU_LINUX
elif [ "$platform" = "linux-gpu" ]
then
  target=$GPU_LINUX
elif [ "$platform" = "darwin" ]
then
  target=$CPU_DARWIN
else
  echo "Please submit a valid platform"
  exit 1
fi

TARGET_DIRECTORY="deps/tensorflow/"
LIBTENSORFLOW="lib/libtensorflow.so"

# Ensure that at least libtensorflow.so is downloaded.
if [ ! -e "$TARGET_DIRECTORY${LIBTENSORFLOW}" ]
then
  curl -L \
    $target |
    tar -C $TARGET_DIRECTORY -xz
fi
