
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 * This differs from util.assertShapesMatch in that it allows values of
 * negative one, an undefined size of a dimensinon, in a shape to match
 * anything.
 */

import {util} from '@tensorflow/tfjs-core';

export function assertShapesMatchAllowUndefinedSize(
    shapeA: number[], shapeB: number[], errorMessagePrefix = ''): void {
  util.assert(
      shapesEqualAllowUndefinedSize(shapeA, shapeB),
      () => errorMessagePrefix + ` Shapes ${shapeA} and ${shapeB} must match`);
}

export function shapesEqualAllowUndefinedSize(n1: number[], n2: number[]) {
  if (n1.length !== n2.length) {
    return false;
  }
  for (let i = 0; i < n1.length; i++) {
    if (n1[i] !== -1 && n2[i] !== -1 && n1[i] !== n2[i]) {
      return false;
    }
  }
  return true;
}
