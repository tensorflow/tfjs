/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as tf from '../../index';
import {describeWithFlags} from '../../jasmine_util';
import {Tensor} from '../../tensor';
import {expectArraysClose} from '../../test_util';
import {WEBGL_ENVS} from './backend_webgl_test_registry';

describeWithFlags('custom-op webgl', WEBGL_ENVS, () => {
  class SquareAndAddKernel implements tf.webgl.GPGPUProgram {
    variableNames = ['X'];
    outputShape: number[];
    userCode: string;
    constructor(inputShape: number[]) {
      this.outputShape = inputShape.slice();

      this.userCode = `
          void main() {
            float x = getXAtOutCoords();
            float value = x * x + x;
            setOutput(value);
          }
        `;
    }
  }

  class SquareAndAddBackpropKernel implements tf.webgl.GPGPUProgram {
    variableNames = ['X'];
    outputShape: number[];
    userCode: string;
    constructor(inputShape: number[]) {
      this.outputShape = inputShape.slice();

      this.userCode = `
          void main() {
            float x = getXAtOutCoords();
            float value = 2.0 * x + 1.0;
            setOutput(value);
          }
        `;
    }
  }

  function squareAndAdd<T extends tf.Tensor>(x: T): T {
    const fn = tf.customGrad((x: tf.Tensor, save: tf.GradSaveFunc) => {
      save([x]);
      const webglBackend = tf.backend() as tf.webgl.MathBackendWebGL;
      const program = new SquareAndAddKernel(x.shape);
      const backpropProgram = new SquareAndAddBackpropKernel(x.shape);

      const value = webglBackend.compileAndRun(program, [x]) as tf.Tensor;

      const gradFunc = (dy: T, saved: Tensor[]) => {
        const [x] = saved;
        return (webglBackend.compileAndRun(backpropProgram, [x]) as T).mul(dy);
      };
      return {value, gradFunc};
    });
    return fn(x) as T;
  }

  it('lets users use custom operations', async () => {
    const inputArr = [1, 2, 3, 4];
    const input = tf.tensor(inputArr);
    const output = squareAndAdd(input);
    expectArraysClose(await output.data(), inputArr.map(x => x * x + x));
  });

  it('lets users define gradients for operations', async () => {
    const inputArr = [1, 2, 3, 4];
    const input = tf.tensor(inputArr);
    const grads = tf.valueAndGrad(x => squareAndAdd(x));
    const {value, grad} = grads(input);
    expectArraysClose(await value.data(), inputArr.map(x => x * x + x));
    expectArraysClose(await grad.data(), inputArr.map(x => 2 * x + 1));
  });

  it('multiplies by dy parameter when it is passed', async () => {
    const inputArr = [1, 2, 3, 4];
    const input = tf.tensor(inputArr);
    const grads = tf.valueAndGrad(x => squareAndAdd(x));
    const {value, grad} = grads(input, tf.zerosLike(input));
    expectArraysClose(await value.data(), inputArr.map(x => x * x + x));
    expectArraysClose(await grad.data(), inputArr.map(() => 0.0));
  });
});
