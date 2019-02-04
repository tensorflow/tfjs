# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

# Before you run this script, do:
# 1) Update the version in package.json
# 2) Run ./scripts/make-version from the base dir of the project.
# 3) Run `yarn` to update `yarn.lock`, in case you updated dependencies
# 4) Commit to the master branch.

# Then:
# 5) Checkout the master branch of this repo.
# 6) Run this script as `./scripts/publish-npm.sh` from the project base dir.

set -e

BRANCH=`git rev-parse --abbrev-ref HEAD`
ORIGIN=`git config --get remote.origin.url`

if [ "$BRANCH" != "master" ]; then
  echo "Error: Switch to the master branch before publishing."
  exit
fi

if ! [[ "$ORIGIN" =~ tensorflow/tfjs-layers ]]; then
  echo "Error: Switch to the main repo (tensorflow/tfjs-layers)."
  exit
fi

yarn build-npm
./scripts/make-version # This is for safety in case you forgot to do 2).
./scripts/tag-version
npm publish --tag next # Remove --tag next when prereleases are done.
echo 'Yay! Published a new package to npm.'
