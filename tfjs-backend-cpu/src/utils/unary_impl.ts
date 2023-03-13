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

import {util} from '@tensorflow/tfjs-core';

import {SimpleUnaryImpl, SimpleUnaryOperation} from './unary_types';

/**
 * Template that creates implementation for unary op.
 */
export function createSimpleUnaryImpl<I extends number | string = number,
  O extends number | string = number>(op: SimpleUnaryOperation<I, O>):
    SimpleUnaryImpl<I, O> {
  return (values, dtype, attrs) => {
    const newValues =
        util.getArrayFromDType(dtype, values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = op(values[i], attrs);
    }
    return newValues;
  };
}
