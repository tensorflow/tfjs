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

yarn prep
rm -rf "$TMP_DIR"
mkdir "$TMP_DIR"

cp -r "docs" "$TMP_DIR/"

# Make the documentation.
./node_modules/.bin/typedoc --out "$TMP_DIR/docs/api/" --excludeExternals \
  --excludeNotExported --excludePrivate --mode file --tsconfig tsconfig-doc.json

# Make demo directory (if not existing)
mkdir -p "$TMP_DIR/demos"

# Copy only top-level files in demos/ to tmp directory (ignore folders)
find demos -maxdepth 1 -type f | xargs -I {} cp {} "$TMP_DIR/demos"
# ... and the images folder.
cp -r demos/images "$TMP_DIR/demos/"

# Copy demo-related tutorials.
demo_tutorials=(
  "mnist"
  "complementary-color-prediction"
  "lstm"
  "rune_recognition"
)
for demo in ${demo_tutorials[@]}
do
  cp -r "demos/$demo" "$TMP_DIR/demos"
done

# Build and copy polymer demos (deploy-demo vulcanizes polymer apps).
polymerdemos=(
  "model-builder"
  "imagenet"
  "nn-art"
  "benchmarks"
  "performance_rnn"
  "teachable_gaming"
  "playground"
  "intro"
  "ml_beginners"
)

# Loop over each demo, copy and build it
for demo in ${polymerdemos[@]}
do
  cp -r "demos/$demo" "$TMP_DIR/demos"
  ./scripts/deploy-demo "demos/$demo" $TMP_DIR
done

# Build vuejs demos.
cd demos/
./node_modules/.bin/poi build vue-demo/ -d "$TMP_DIR/demos/vue-demo/"
./node_modules/.bin/poi build latent-space-explorer/ \
  -d "$TMP_DIR/demos/latent-space-explorer/"
cd ..

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

bundle install
bundle exec jekyll serve
