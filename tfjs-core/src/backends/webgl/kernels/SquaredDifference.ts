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

import {env} from '../../../environment';
import {SquaredDifference, SquaredDifferenceInputs} from '../../../kernel_names';
import {KernelConfig} from '../../../kernel_registry';
import {MathBackendWebGL} from '../backend_webgl';
import {BinaryOpProgram} from '../binaryop_gpu';
import {BinaryOpPackedProgram} from '../binaryop_packed_gpu';

export const squaredDifferenceConfig: KernelConfig = {
  kernelName: SquaredDifference,
  backendName: 'webgl',
  kernelFunc: ({inputs, backend}) => {
    const {a, b} = inputs as SquaredDifferenceInputs;
    const SQUARED_DIFFERENCE = 'return (a - b) * (a - b);';
    const webGLBackend = backend as MathBackendWebGL;

    const program = env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
        new BinaryOpPackedProgram(SQUARED_DIFFERENCE, a.shape, b.shape) :
        new BinaryOpProgram(SQUARED_DIFFERENCE, a.shape, b.shape);
    return webGLBackend.compileAndRun(program, [a, b]);
  }
};
