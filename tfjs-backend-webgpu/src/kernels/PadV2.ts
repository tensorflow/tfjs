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

import {KernelConfig, KernelFunc, PadV2, PadV2Attrs, PadV2Inputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {identity} from './Identity';
import {PadProgram} from './pad_webgpu';
import {fill} from './Fill';

export const padV2 =
    (args: {inputs: PadV2Inputs,
            backend: WebGPUBackend,
            attrs: PadV2Attrs}): TensorInfo => {
      const {inputs, backend, attrs} = args;
      const {x} = inputs;
      const {paddings, constantValue} = attrs;
      if (paddings.every(p => util.arraysEqual(p, [0, 0]))) {
        return identity({inputs: {x}, backend});
      }
      if (util.sizeFromShape(x.shape) === 0) {
        // Short-circuit the computation, since x doesn't have value, only
        // the shape is used to compute output shape to pad.
        const outputShape = paddings.map(
            (p, i) =>
                p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);
        return fill({
          backend,
          attrs: {shape: outputShape, value: constantValue, dtype: x.dtype}
        });
      }
      const uniformData = [{type: 'float32', data: [constantValue]}];
      paddings.map(p => uniformData.push({type: 'int32', data: [p[0], p[1]]}));
      const program = new PadProgram(x.shape, paddings);
      return backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
    };

export const padV2Config: KernelConfig = {
  kernelName: PadV2,
  backendName: 'webgpu',
  kernelFunc: padV2 as {} as KernelFunc
};
