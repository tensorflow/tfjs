/**
 * @license
 * Copyright 2023 Google LLC.
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

import * as tf from '@tensorflow/tfjs-core';
import {WebGPUBackend} from './backend_webgpu';
import {describeWebGPU} from './test_util';
import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

class InvalidShaderProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A'];
  workgroupSize: [number, number, number];
  size = true;

  constructor(outputShape: number[]) {
    const workgroupSizeX = 128;
    this.workgroupSize = [workgroupSizeX, 1, 1];
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);
    this.shaderKey = 'invalidShader';
  }

  getUserCode(): string {
    return `
      ${main('index')} {
        if (index < uniforms.size) {
          let a = getAByOutputIndex(index);
          setOutputAtIndex(index, a-);
        }
      }
      `;
  }
}

function invalidShader<T extends tf.Tensor>(x: T): T {
  const webglBackend = tf.backend() as WebGPUBackend;
  const program = new InvalidShaderProgram(x.shape);

  const outInfo: tf.TensorInfo =
      webglBackend.runWebGPUProgram(program, [x], x.dtype);
  const value = tf.engine().makeTensorFromTensorInfo(outInfo) as T;

  return value;
}

describeWebGPU('invalid webgpu shader', () => {
  let prevBackend: string;
  let savedWebGPUCPUForward: number|boolean;
  let savedEngineCompileOnly: number|boolean;
  let webGPUBackend: WebGPUBackend;
  const customWebGPUBackendName = 'invalid-shader';

  beforeAll(() => {
    prevBackend = tf.getBackend();
  });

  beforeEach(async () => {
    const adapter = await navigator.gpu.requestAdapter({});
    const device = await adapter.requestDevice({});
    webGPUBackend = new WebGPUBackend(device);

    tf.copyRegisteredKernels('webgpu', customWebGPUBackendName);
    tf.registerBackend(customWebGPUBackendName, () => webGPUBackend);
    tf.setBackend(customWebGPUBackendName);

    savedWebGPUCPUForward = tf.env().get('WEBGPU_CPU_FORWARD');
    savedEngineCompileOnly = tf.env().get('ENGINE_COMPILE_ONLY');
    tf.env().set('WEBGPU_CPU_FORWARD', false);
    await tf.ready();
  });

  afterEach(() => {
    tf.env().set('WEBGPU_CPU_FORWARD', savedWebGPUCPUForward);
    tf.env().set('ENGINE_COMPILE_ONLY', savedEngineCompileOnly);
    tf.setBackend(prevBackend);
    tf.removeBackend(customWebGPUBackendName);
  });

  it('throw error when compile invalid shader parallelly', async () => {
    const input = tf.tensor([1, 2, 3, 4]);
    // Parallel compile.
    tf.env().set('ENGINE_COMPILE_ONLY', true);
    invalidShader(input);
    await expectAsync(
        (tf.backend() as WebGPUBackend).checkCompileCompletionAsync())
        .toBeRejectedWith(new Error(
            `[Invalid ShaderModule "InvalidShaderProgram"] is invalid.`));
  });
});
