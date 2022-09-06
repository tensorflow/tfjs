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

import {BatchMatMul, BatchMatMulAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {matMul} from '../ops/mat_mul';
import {Tensor} from '../tensor';

export const batchMatMulGradConfig: GradConfig = {
  kernelName: BatchMatMul,
  inputsToSave: ['a', 'b'],
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const [a, b] = saved;

    const {transposeA, transposeB} = attrs as {} as BatchMatMulAttrs;

    if (!transposeA && !transposeB) {
      return {
        a: () => matMul(dy, b, false, true),
        b: () => matMul(a, dy, true, false)
      };
    } else if (!transposeA && transposeB) {
      return {
        a: () => matMul(dy, b, false, false),
        b: () => matMul(dy, a, true, false)
      };
    } else if (transposeA && !transposeB) {
      return {
        a: () => matMul(b, dy, false, true),
        b: () => matMul(a, dy, false, false)
      };
    } else {
      return {
        a: () => matMul(b, dy, true, true),
        b: () => matMul(dy, a, true, true)
      };
    }
  }
};
