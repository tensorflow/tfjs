/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
import {Concat, ConcatAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {split} from '../ops/split';
import {Tensor} from '../tensor';
import {parseAxisParam} from '../util';

export const concatGradConfig: GradConfig = {
  kernelName: Concat,
  saveAllInputs: true,
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const shapes = saved.map(t => t.shape);
    const {axis} = attrs as {} as ConcatAttrs;
    const $axis = parseAxisParam(axis, saved[0].shape)[0];
    const sizeSplits = shapes.map(s => s[$axis]);
    const derTensors = split(dy, sizeSplits, $axis);
    return derTensors.map(t => () => t) as {};
  }
};
