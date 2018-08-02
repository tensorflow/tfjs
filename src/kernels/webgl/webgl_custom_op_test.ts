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
import {expectArraysClose, WEBGL_ENVS} from '../../test_util';

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
    const fn = tf.customGrad(x => {
      const webglBackend = tf.ENV.backend as tf.webgl.MathBackendWebGL;
      const program = new SquareAndAddKernel(x.shape);
      const backpropProgram = new SquareAndAddBackpropKernel(x.shape);

      const value = webglBackend.compileAndRun(program, [x]);

      const gradFunc = (dy: T) =>
          webglBackend.compileAndRun(backpropProgram, [x]).mul(dy) as T;
      return {value, gradFunc};
    });
    return fn(x) as T;
  }

  it('lets users use custom operations', () => {
    const inputArr = [1, 2, 3, 4];
    const input = tf.tensor(inputArr);
    const output = squareAndAdd(input);
    expectArraysClose(output, inputArr.map(x => x * x + x));
  });

  it('lets users define gradients for operations', () => {
    const inputArr = [1, 2, 3, 4];
    const input = tf.tensor(inputArr);
    const grads = tf.valueAndGrad(x => squareAndAdd(x));
    const {value, grad} = grads(input);
    expectArraysClose(value, inputArr.map(x => x * x + x));
    expectArraysClose(grad, inputArr.map(x => 2 * x + 1));
  });

  it('multiplies by dy parameter when it is passed', () => {
    const inputArr = [1, 2, 3, 4];
    const input = tf.tensor(inputArr);
    const grads = tf.valueAndGrad(x => squareAndAdd(x));
    const {value, grad} = grads(input, tf.zerosLike(input));
    expectArraysClose(value, inputArr.map(x => x * x + x));
    expectArraysClose(grad, inputArr.map(() => 0.0));
  });
});
