#!/bin/bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# ==============================================================================


# This script applies google-style clang format on all TypeScript (.ts) files
# within a certain scope.
#
# Usage examples:
# 1. Format all .ts files touched by this change (unstaged or staged in git).
#   clang_format_ts.sh
#
# 2. Format all .ts files under the source tree.
#   clang_format_ts.sh -a
#
# 3. Format specific files.
#   clang_format_ts.sh src/types.ts

set -e

FILE_SCOPE=""

if [[ "$#" -gt 0 ]]; then
  while true; do
    if [[ -z "$1" ]]; then
      break
    fi
    if [[ "$1" == "-a" ]]; then
      if [[ -z "${FILE_SCOPE}" ]]; then
        FILE_SCOPE="__all__"
      else
        echo "ERROR: -a flag should not be used with file names"
        exit 1
      fi
    else
      if [[ "${FILE_SCOPE}" != "__all__" ]]; then
        FILE_SCOPE="${FILE_SCOPE} $1"
      else
        echo "ERROR: -a flag should not be used with file names"
        exit 1
      fi
    fi
    shift
  done
else
  FILE_SCOPE="__touched__"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLANG_FORMAT_PREFIX="clang-format -i --verbose --style=google"
if [[ "${FILE_SCOPE}" == "__touched__" ]]; then
  TOUCHED_TS_FILES="$(git status --porcelain | grep '.*\.ts$' | sed s/^...//)"

  if [[ -z "${TOUCHED_TS_FILES}" ]]; then
    exit 0
  else
    pushd "${SCRIPT_DIR}/.." > /dev/null
    for TS_FILE in ${TOUCHED_TS_FILES}; do
      if [[ -f ${TS_FILE} ]]; then
        ${CLANG_FORMAT_PREFIX} "${TS_FILE}"
      fi
    done
    popd > /dev/null
  fi
elif [[ "${FILE_SCOPE}" == "__all__" ]]; then
  ALL_TS_FILES="$(find "${SCRIPT_DIR}/../src" "${SCRIPT_DIR}/../demos" -name '*.ts')"
  for TS_FILE in ${ALL_TS_FILES}; do
    ${CLANG_FORMAT_PREFIX} "${TS_FILE}"
  done

else
  ${CLANG_FORMAT_PREFIX} ${FILE_SCOPE}
fi
