/**
 * @license
 * Copyright 2023 Google LLC.
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

import { Tensor } from '@tensorflow/tfjs-core';

export function tensorToArr(input: Tensor): unknown[] {
  return Array.from(input.dataSync()) as unknown as unknown[];
}

export function tensorArrTo2DArr(inputs: Tensor[]): unknown[][] {
  return inputs.map(input => tensorToArr(input));
}
