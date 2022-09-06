/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * Generates sparse fill empty rows indices, dense shape mismatch error message.
 *
 * @param indicesLength The first dimension of indices.
 */
export function getSparseFillEmptyRowsIndicesDenseShapeMismatch(
    indicesLength: number) {
  return `Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${indicesLength}`;
}

/**
 * Generates sparse fill empty rows negative index error message.
 *
 * @param index The index with a negative value.
 * @param value The negative value.
 */
export function getSparseFillEmptyRowsNegativeIndexErrorMessage(
    index: number, value: number) {
  return `indices(${index}, 0) is invalid: ${value} < 0`;
}

/**
 * Generates sparse fill empty rows out of range index error message.
 *
 * @param index The index with an out of range value.
 * @param value The out of range value.
 * @param limit The upper limit for indices.
 */
export function getSparseFillEmptyRowsOutOfRangeIndexErrorMessage(
    index: number, value: number, limit: number) {
  return `indices(${index}, 0) is invalid: ${value} >= ${limit}`;
}
