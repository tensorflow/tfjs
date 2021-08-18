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

import {KernelConfig, Tensor4D} from '@tensorflow/tfjs-core';
import {FlipLeftRight, FlipLeftRightInputs} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {FlipLeftRightProgram} from '../flip_left_right_gpu';

export const flipLeftRightConfig: KernelConfig = {
  kernelName: FlipLeftRight,
  backendName: 'webgl',
  kernelFunc: ({inputs, backend}) => {
    const {image} = inputs as FlipLeftRightInputs;
    const webglBackend = backend as MathBackendWebGL;

    const program = new FlipLeftRightProgram((image as Tensor4D).shape);
    const output = webglBackend.runWebGLProgram(program, [image], image.dtype);
    return output;
  }
};
