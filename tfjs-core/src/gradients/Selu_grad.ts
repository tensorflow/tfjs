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
import {Selu} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import {cast} from '../ops/cast';
import {exp} from '../ops/exp';
import {greater} from '../ops/greater';
import {mul} from '../ops/mul';
import {scalar} from '../ops/scalar';
import {SELU_SCALE, SELU_SCALEALPHA} from '../ops/selu_util';
import {where} from '../ops/where';
import {Tensor} from '../tensor';

export const seluGradConfig: GradConfig = {
  kernelName: Selu,
  inputsToSave: ['x'],
  gradFunc: (dy: Tensor, saved: Tensor[]) => {
    const [x] = saved;
    return {
      x: () => {
        const mask = greater(x, scalar(0));

        const scaleAlpha = scalar(SELU_SCALEALPHA);
        const scale = scalar(SELU_SCALE);

        const greaterThanZeroDer = mul(dy, scale);
        const lessEqualZeroDer =
            mul(mul(dy, scaleAlpha), exp(cast(x, 'float32')));

        return where(mask, greaterThanZeroDer, lessEqualZeroDer);
      }
    };
  }
};
