# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

# Before you run this script, do this:
# 1) Update the version in package.json
# 2) Run ./scripts/make-version from the base dir of the project.
# 3) Commit to the master branch.

# Then:
# 4) Run this script as `yarn publish-npm` from the base dir of the project.

set -e

yarn build-npm
./scripts/make-version # This is for safety in case you forgot to do 2).
npm publish
./scripts/tag-version
echo 'Yay! Published a new package to npm.'
