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

# Builds the website and stages it for preview on localhost:4000. If all is good
# run ./script/publish-website.sh to deploy it to https://deeplearnjs.org.

TMP_DIR="/tmp/deeplearn-website"

npm run prep
rm -rf "$TMP_DIR"
mkdir "$TMP_DIR"

cp -r "docs" "$TMP_DIR/"

# Make the documentation.
./node_modules/.bin/typedoc --out "$TMP_DIR/docs/api/" --excludeExternals \
  --excludeNotExported --excludePrivate --mode file --tsconfig tsconfig-doc.json

# Build the demos (deploy-demo vulcanizes polymer apps).
cp -r "demos" "$TMP_DIR/"
./scripts/deploy-demo demos/model-builder $TMP_DIR
./scripts/deploy-demo demos/imagenet $TMP_DIR
./scripts/deploy-demo demos/nn-art $TMP_DIR
./scripts/deploy-demo demos/benchmarks $TMP_DIR
./scripts/deploy-demo demos/performance_rnn $TMP_DIR
./scripts/deploy-demo demos/teachable_gaming $TMP_DIR
./scripts/deploy-demo demos/playground $TMP_DIR

./scripts/deploy-demo demos/intro $TMP_DIR
./scripts/deploy-demo demos/ml_beginners $TMP_DIR

# Build the homepage (no deploy since homepage is not polymer).
./scripts/build-demo demos/homepage
cp -r demos/homepage/* "$TMP_DIR"
cp "README.md" "$TMP_DIR/_includes/"
rm "$TMP_DIR"/homepage.ts

echo "Website staged at $TMP_DIR"
pushd $TMP_DIR > /dev/null

if ! [ -x "$(command -v bundle)" ]; then
  echo 'Installing Bundler'
  gem install bundler
fi

bundle install --clean
bundle exec jekyll serve
