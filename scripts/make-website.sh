#!/usr/bin/env bash
# Copyright 2017 Google Inc. All Rights Reserved.
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
TMP_DIR="/tmp/deeplearn-website"

npm run prep
rm -rf "$TMP_DIR"
mkdir "$TMP_DIR"

cp -r "docs" "$TMP_DIR/"
cp "README.md" "$TMP_DIR/"

# Make the documentation.
./node_modules/.bin/typedoc --out "$TMP_DIR/docs/api/" --excludeExternals \
  --excludeNotExported --excludePrivate --mode file --tsconfig tsconfig-doc.json

# Build the demos (deploy-demo vulcanizes polymer apps).
cp -r "demos" "$TMP_DIR/"
./scripts/deploy-demo demos/model-builder/model-builder.ts \
    demos/model-builder/model-builder-demo.html $TMP_DIR/demos/model-builder/
./scripts/deploy-demo demos/imagenet/imagenet-demo.ts \
    demos/imagenet/imagenet-demo.html $TMP_DIR/demos/imagenet
./scripts/deploy-demo demos/nn-art/nn-art.ts \
    demos/nn-art/nn-art-demo.html $TMP_DIR/demos/nn-art
./scripts/deploy-demo demos/benchmarks/math-benchmark.ts \
    demos/benchmarks/benchmark-demo.html $TMP_DIR/demos/benchmarks

# Build the homepage (no deploy since homepage is not polymer).
./scripts/build-demo demos/homepage/index.ts
cp -r demos/homepage/* "$TMP_DIR"
rm "$TMP_DIR"/index.ts

git stash
git checkout gh-pages

cp -rf "$TMP_DIR"/* .

git add .
git commit -m "github pages"

git checkout master
rm -f -r "_site/"
git stash pop
