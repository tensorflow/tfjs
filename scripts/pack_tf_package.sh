#!/usr/bin/env bash
# Copyright 2017 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# Copy the output files from compiling tensorflow to a single
# publishable tar file.

set -e
mkdir tf_build

mkdir tf_build/lib
cp -P ../tensorflow/bazel-bin/tensorflow/libtensorflow*.so* tf_build/lib

mkdir -p tf_build/include/tensorflow/core/platform
mkdir -p tf_build/include/tensorflow/c/eager

cp ../tensorflow/tensorflow/core/platform/ctstring.h tf_build/include/tensorflow/core/platform/
cp ../tensorflow/tensorflow/core/platform/ctstring_internal.h tf_build/include/tensorflow/core/platform/

cp ../tensorflow/tensorflow/c/c_api.h tf_build/include/tensorflow/c/
cp ../tensorflow/tensorflow/c/c_api_experimental.h tf_build/include/tensorflow/c/
cp ../tensorflow/tensorflow/c/c_api_macros.h tf_build/include/tensorflow/c/
cp ../tensorflow/tensorflow/c/tensor_interface.h tf_build/include/tensorflow/c/
cp ../tensorflow/tensorflow/c/tf_attrtype.h tf_build/include/tensorflow/c/
cp ../tensorflow/tensorflow/c/tf_datatype.h tf_build/include/tensorflow/c/
cp ../tensorflow/tensorflow/c/tf_file_statistics.h tf_build/include/tensorflow/c/
cp ../tensorflow/tensorflow/c/tf_status.h tf_build/include/tensorflow/c/
cp ../tensorflow/tensorflow/c/tf_tensor.h tf_build/include/tensorflow/c/
cp ../tensorflow/tensorflow/c/tf_tstring.h tf_build/include/tensorflow/c/

cp ../tensorflow/tensorflow/c/eager/c_api.h tf_build/include/tensorflow/c/eager/
cp ../tensorflow/tensorflow/c/eager/c_api_experimental.h tf_build/include/tensorflow/c/eager/
cp ../tensorflow/tensorflow/c/eager/c_api_unified_experimental.h tf_build/include/tensorflow/c/eager/
cp ../tensorflow/tensorflow/c/eager/dlpack.h tf_build/include/tensorflow/c/eager/

cd tf_build
tar czf libtensorflow_r2_8_linux_arm64.tar.gz ./lib ./include
