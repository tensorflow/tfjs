import {TypedArray} from '../types';

/*
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 * Tests a boolean expression and throws a message if false.
 */
export function assert(expr: boolean, msg: string|(() => string)) {
  if (!expr) {
    throw new Error(typeof msg === 'string' ? msg : msg());
  }
}

export function assertShapesMatch(
    shapeA: number[], shapeB: number[], errorMessagePrefix = ''): void {
  assert(
      arraysEqual(shapeA, shapeB),
      errorMessagePrefix + ` Shapes ${shapeA} and ${shapeB} must match`);
}

export function arraysEqual(n1: number[]|TypedArray, n2: number[]|TypedArray) {
  if (n1.length !== n2.length) {
    return false;
  }
  for (let i = 0; i < n1.length; i++) {
    if (n1[i] !== n2[i]) {
      return false;
    }
  }
  return true;
}

// Number of decimal places to  when checking float similarity
export const DECIMAL_PLACES_TO_CHECK = 4;
