
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
    shapeA: number|number[], shapeB: number|number[],
    errorMessagePrefix = ''): void {
  // constant shape means unknown rank
  if (typeof shapeA === 'number' || typeof shapeB === 'number') {
    return;
  }
  util.assert(
      shapeA.length === shapeB.length,
      () => errorMessagePrefix + ` Shapes ${shapeA} and ${shapeB} must match`);
  for (let i = 0; i < shapeA.length; i++) {
    const dim0 = shapeA[i];
    const dim1 = shapeB[i];
    util.assert(
        dim0 < 0 || dim1 < 0 || dim0 === dim1,
        () =>
            errorMessagePrefix + ` Shapes ${shapeA} and ${shapeB} must match`);
  }
}
