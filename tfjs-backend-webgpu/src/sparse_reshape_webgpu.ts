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

import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class SparseReshapeOutputShapeProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['inputShape', 'newShape'];
  workgroupSize: [number, number, number] = [1, 1, 1];

  constructor(outShape: number[]) {
    this.outputShape = outShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = [1, 1, 1];
    this.shaderKey = 'sparseOutputShape';
  }

  getUserCode(): string {
    const userCode = `
      ${main('index')} {
        var denseSize = 1;
        for (var i = 0; i < uniforms.inputShapeShape; i++) {
          denseSize *= inputShape[i];
        }

        var product = 1;
        var unknownIndex = -1;
        for (var i = 0; i < uniforms.newShapeShape; i++) {
          let shape = newShape[i];
          if (shape == -1) {
            unknownIndex = i;
          } else {
            product *= shape;
            result[i] = shape;
          }
        }

        if (unknownIndex != -1) {
          result[unknownIndex] = denseSize / product;
        }
      }
    `;
    return userCode;
  }
}

export class SparseReshapeOutputIndicesProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['inputIndices', 'inputShape', 'newShape'];
  workgroupSize: [number, number, number] = [64, 1, 1];
  size = true;

  constructor(outShape: number[]) {
    this.outputShape = outShape;
    this.dispatchLayout = flatDispatchLayout([this.outputShape[0]]);
    this.dispatch = computeDispatch(
        this.dispatchLayout, [this.outputShape[0]], this.workgroupSize);
    this.shaderKey = 'sparseReshapeOutputIndices';
  }

  getUserCode(): string {
    const userCode = `
      var<workgroup> inputStrides : array<i32, 6>;
      var<workgroup> outputStrides : array<i32, 6>;
      ${main('index')} {
        if (localId.x == 0) {
          inputStrides[uniforms.inputShapeShape - 1] = 1;
          for (var i = uniforms.inputShapeShape - 2; i >= 0; i--) {
            inputStrides[i] = inputStrides[i + 1] * inputShape[i + 1];
          }

          outputStrides[uniforms.newShapeShape - 1] = 1;
          for (var i = uniforms.newShapeShape - 2; i >= 0; i--) {
            outputStrides[i] = outputStrides[i + 1] * newShape[i + 1];
          }
        }
        workgroupBarrier();

        if (index < uniforms.size) {
          var id = 0;
          for (var i = 0; i < uniforms.inputShapeShape; i++) {
            id += inputIndices[index * uniforms.inputShapeShape + i] * inputStrides[i];
          }

          for (var i = 0; i < uniforms.newShapeShape; i++) {
            result[index * uniforms.newShapeShape + i] = id / outputStrides[i];
            id %= outputStrides[i];
          }
        }
      }
    `;
    return userCode;
  }
}
