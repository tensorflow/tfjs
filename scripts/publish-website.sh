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

# Deploys the staged website to https://deeplearnjs.org.
# Before running this script, run ./scripts/make-website.sh which stages the
# website and serves it on localhost:4000 for preview.

TMP_DIR="/tmp/deeplearn-website"
pushd $TMP_DIR > /dev/null
bundle exec jekyll build
pushd $TMP_DIR/_site > /dev/null
gsutil -m rsync -d -r . gs://deeplearnjs.org
gsutil -m setmeta -h "Cache-Control:private" "gs://deeplearnjs.org/**.html"
gsutil -m setmeta -h "Cache-Control:private" "gs://deeplearnjs.org/**.css"
gsutil -m setmeta -h "Cache-Control:private" "gs://deeplearnjs.org/**.js"
popd > /dev/null
popd > /dev/null
