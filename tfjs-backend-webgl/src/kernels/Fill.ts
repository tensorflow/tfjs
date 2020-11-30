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

import {Fill, FillAttrs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {FillProgram} from '../fill_gpu';

export function fill(args: {backend: MathBackendWebGL, attrs: FillAttrs}):
    TensorInfo {
  const {backend, attrs} = args;
  const {shape, value} = attrs;
  let {dtype} = attrs;

  dtype = dtype || util.inferDtype(value);

  if (dtype === 'string') {
    // String type should be handled in CPU memory.
    const values = util.getArrayFromDType(dtype, util.sizeFromShape(shape));
    values.fill(value as string);
    return backend.makeTensorInfo(shape, dtype, values);
  } else {
    const program = new FillProgram(shape, value as number);
    const customSetup = program.getCustomSetupFunc(value as number);
    return backend.runWebGLProgram(program, [], dtype, customSetup);
  }
}

export const fillConfig: KernelConfig = {
  kernelName: Fill,
  backendName: 'webgl',
  kernelFunc: fill as {} as KernelFunc
};
