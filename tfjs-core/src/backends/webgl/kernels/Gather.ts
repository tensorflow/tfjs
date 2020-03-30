/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {Gather, GatherAttrs, GatherInputs} from '../../../kernel_names';
import {KernelConfig} from '../../../kernel_registry';
import {Tensor1D} from '../../../tensor';
import {MathBackendWebGL} from '../backend_webgl';
import {GatherProgram} from '../gather_gpu';

export const gatherConfig: KernelConfig = {
  kernelName: Gather,
  backendName: 'webgl',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {x, indices} = inputs as GatherInputs;
    const {axis} = attrs as unknown as GatherAttrs;
    const webglBackend = backend as MathBackendWebGL;
    const program =
        new GatherProgram(x.shape, (indices as Tensor1D).size, axis);
    return webglBackend.runWebGLProgram(
        program, [x, (indices as Tensor1D).flatten()], x.dtype);
  }
};
