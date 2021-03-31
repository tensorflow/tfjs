#!/bin/bash

set -e

"$(npm bin)/tsc" --noEmit --watch &
"$(npm bin)/parcel" ./src/index.html
# "$(npm bin)/parcel" ./src/index.html
