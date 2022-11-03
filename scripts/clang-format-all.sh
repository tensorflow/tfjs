#!/usr/bin/env bash
# Copyright 2022 Google LLC.
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

# Format all js, ts, jsx, and tsx files with clang-format.

set -eEuo pipefail
cd `git rev-parse --show-toplevel`

# `-print` avoids printing `node_modules` and `dist` directories that were pruned.
FILES=`find . -type d -name node_modules -prune -o -type d -name dist -prune -o \( -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.jsx" \) -print`

parallel ./node_modules/.bin/clang-format -i --style='file' -- $FILES
