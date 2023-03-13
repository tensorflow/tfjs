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

import {Minimum} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import {cast} from '../ops/cast';
import {greater} from '../ops/greater';
import {lessEqual} from '../ops/less_equal';
import {mul} from '../ops/mul';
import {Tensor} from '../tensor';

export const minimumGradConfig: GradConfig = {
  kernelName: Minimum,
  inputsToSave: ['a', 'b'],
  gradFunc: (dy: Tensor, saved: Tensor[]) => {
    const [a, b] = saved;
    const derA = () => mul(dy, cast(lessEqual(a, b), 'float32'));
    const derB = () => mul(dy, cast(greater(a, b), 'float32'));
    return {a: derA, b: derB};
  }
};
