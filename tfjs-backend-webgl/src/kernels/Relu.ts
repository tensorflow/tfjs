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

import {KernelConfig, Relu, ReluInputs} from '@tensorflow/tfjs-core';
import {env} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {RELU, UnaryOpProgram} from '../unaryop_gpu';
import * as unaryop_packed_gpu from '../unaryop_packed_gpu';
import {UnaryOpPackedProgram} from '../unaryop_packed_gpu';

export const reluConfig: KernelConfig = {
  kernelName: Relu,
  backendName: 'webgl',
  kernelFunc: ({inputs, backend}) => {
    const {x} = inputs as ReluInputs;
    const webglBackend = backend as MathBackendWebGL;

    let program = new UnaryOpProgram(x.shape, RELU);
    if (env().getBool('WEBGL_PACK')) {
      program = new UnaryOpPackedProgram(x.shape, unaryop_packed_gpu.RELU);
    }
    return webglBackend.runWebGLProgram(program, [x], x.dtype);
  }
};
