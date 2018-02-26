#!/bin/sh

CPU_DARWIN="http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow/TYPE=mac-slave/lastSuccessfulBuild/artifact/lib_package/libtensorflow-cpu-darwin-x86_64.tar.gz"
GPU_LINUX="http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow/TYPE=gpu-linux/lastSuccessfulBuild/artifact/lib_package/libtensorflow-gpu-linux-x86_64.tar.gz"
# TODO(kreeger): Use nightly links when Jenkins TF uses an updated bazel.
CPU_LINUX="https://storage.googleapis.com/tf-buiilds/libtensorflow.tar.gz"
# CPU_LINUX="http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow/TYPE=cpu-slave/lastSuccessfulBuild/artifact/lib_package/libtensorflow-cpu-linux-x86_64.tar.gz"

target=""
platform=$1
if [ "$platform" == "linux-cpu" ]
then
  target=$CPU_LINUX
elif [ "$platform" == "linux-gpu" ]
then
  target=$GPU_LINUX
elif [ "$platform" == "darwin" ]
then
  target=$CPU_DARWIN
else
  echo "Please submit a valid platform"
  exit 1
fi

TARGET_DIRECTORY="deps/tensorflow/"

# TODO(kreeger): Drop this when the eager header ships w/ libtensorflow.
eager_include_path="deps/tensorflow/include/tensorflow/c/eager"
mkdir -p $eager_include_path
curl -o \
  $eager_include_path"/c_api.h" \
  "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/c/eager/c_api.h"

curl -L \
  $target |
  tar -C $TARGET_DIRECTORY -xz
