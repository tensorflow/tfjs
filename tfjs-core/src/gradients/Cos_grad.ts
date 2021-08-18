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

import {Cos} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import {cast} from '../ops/cast';
import {mul} from '../ops/mul';
import {neg} from '../ops/neg';
import {sin} from '../ops/sin';
import {Tensor} from '../tensor';

export const cosGradConfig: GradConfig = {
  kernelName: Cos,
  inputsToSave: ['x'],
  gradFunc: (dy: Tensor, saved: Tensor[]) => {
    const [x] = saved;

    return {x: () => mul(neg(sin(cast(x, 'float32'))), dy)};
  }
};
