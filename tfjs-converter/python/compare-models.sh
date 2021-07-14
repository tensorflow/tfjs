#!/usr/bin/env bash
# Copyright 2021 Google LLC
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
# ==============================================================================

# Compare python tensorflow model outputs to TFJS converted model outputs for
# debugging purposes.
#
# Usage:
#   compare-models.sh \
#       <MODEL_DIR> <OUTPUT_PATH> [GRAPH_INPUT_1] [GRAPH_INPUT_2] ...
#
# Positional arguments:
#   MODEL_DIR: Directory where the converted TFJS model is found (as converted
#              by the tensorflowjs_converter command in the
#              tensorflowjs pip package).
#   OUTPUT_PATH: Destination file for outputting model differences.
#
# Optional positional argumnets:
#   GRAPH_INPUT_1, GRAPH_INPUT_2 ... : Inputs that the models will be run
#    run against. These must me listed according to the sorted order of
#    the input names that the inputs correspond to, and in JSON format
set -e

function print_usage() {
  echo "Usage:"
  echo "  compare-models.sh \\"
  echo "      <MODEL_DIR> <OUTPUT_PATH> [GRAPH_INPUT_1] [GRAPH_INPUT_2] ..."
  echo
}

if [[ $# -le 1 ]]; then
  print_usage
  exit 1
fi

ALL_ARGS=("$@")
INPUT_VALUES=("${ALL_ARGS[@]:2}")

TEMP_FILE=$(mktemp)
trap "rm -f $TEMP_FILE" 0 2 3 15

python compare-models.py --model_dir "$1" --get_output_nodes --output_path "$TEMP_FILE"
ts-node --transpile-only -P ../tsconfig.test.json ../src/get_model_outputs.ts\
        --model_dir "$1" --input_values "$INPUT_VALUES" --output_nodes_path "$TEMP_FILE" --output_file "$TEMP_FILE"
python compare-models.py --model_dir "$1" --input_values "$INPUT_VALUES" --tfjs_json_path "$TEMP_FILE" --output_path "$2"
