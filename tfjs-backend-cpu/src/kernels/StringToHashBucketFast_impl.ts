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

import {TypedArray, util} from '@tensorflow/tfjs-core';

export function stringToHashBucketFastImpl(
    input: Uint8Array[], numBuckets: number): TypedArray {
  const output = util.getArrayFromDType('int32', input.length) as TypedArray;

  for (let i = 0; i < input.length; ++i) {
    output[i] =
        util.fingerPrint64(input[i]).modulo(numBuckets).getLowBitsUnsigned();
  }

  return output;
}
