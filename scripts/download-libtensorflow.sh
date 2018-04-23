#!/bin/sh

# Build new package:
# `bazel build //tensorflow/tools/lib_package:libtensorflow`

CPU_DARWIN="https://storage.googleapis.com/tf-builds/libtensorflow_r1_8_darwin.tar.gz"
CPU_LINUX="https://storage.googleapis.com/tf-builds/libtensorflow_r1_8_linux_cpu.tar.gz"

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
