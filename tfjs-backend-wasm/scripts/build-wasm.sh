#!/usr/bin/env bash
# Copyright 2019 Google LLC. All Rights Reserved.
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

set -e

SRCS="src/lib.cc src/kernels.cc"

# Add these optimization flags in production: -g0 -O3 --llvm-lto 3

emcc $SRCS \
  -std=c++11 \
  -fno-rtti \
  -g \
  -fno-exceptions \
  -I./src/ \
  -o wasm-out/tfjs-backend-wasm.js \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s DEFAULT_LIBRARY_FUNCS_TO_INCLUDE=[] \
  -s DISABLE_EXCEPTION_CATCHING=1 \
  -s FILESYSTEM=0 \
  -s EXIT_RUNTIME=0 \
  -s EXPORTED_FUNCTIONS='["_malloc"]' \
  -s EXTRA_EXPORTED_RUNTIME_METHODS='["cwrap"]' \
  -s ENVIRONMENT=web \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=WasmBackendModule \
  -s MALLOC=emmalloc
