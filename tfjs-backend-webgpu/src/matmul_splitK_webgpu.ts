/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {backend_util, TensorInfo, util} from '@tensorflow/tfjs-core';

import {activationFnSnippet, biasActivationSnippet} from './activation_util';
import {getMainHeaderAndGlobalIndexString, getMainHeaderString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class MatMulSplitKProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workGroupSize: [number, number, number] = [8, 8, 1];
  elementsPerThread: [number, number, number];
  transposeA: boolean;
  transposeB: boolean;
  atomic = true;
  batchAEqualOne: boolean;
  batchBEqualOne: boolean;
  tileInner = 32;

  constructor(
      outputShape: [number, number, number], dimInner: number,
      batchAEqualOne: boolean, batchBEqualOne: boolean, transposeA = false,
      transposeB = false) {
    util.assert(
        outputShape[0] === 1,
        () => 'MatMulSplitKProgram only supports batch = 1.');
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0, 3]};
    this.elementsPerThread = [4, 4, this.tileInner];
    if (this.outputShape[1] < 16) {
      this.elementsPerThread[1] = 1;
    }
    if (this.outputShape[2] < 16) {
      this.elementsPerThread[0] = 1;
    }
    this.dispatch = computeDispatch(
        this.dispatchLayout,
        [
          this.outputShape[0], this.outputShape[1], this.outputShape[2],
          dimInner
        ],
        this.workGroupSize, this.elementsPerThread);

    this.transposeA = transposeA;
    this.transposeB = transposeB;
    this.batchAEqualOne = batchAEqualOne;
    this.batchBEqualOne = batchBEqualOne;
    this.shaderKey = `matMulSplitK_${transposeA}_${transposeB}_${
        batchAEqualOne}_${batchBEqualOne}_${this.elementsPerThread}`;
  }

  getUserCode(): string {
    let sampleA;
    if (this.transposeA === false) {
      sampleA =
          `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimAOuter, uniforms.dimInner))) {
            value = A[batch * batchASize + row * uniforms.dimInner + col];
          }
          return value;`;
    } else {
      sampleA =
          `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimAOuter, uniforms.dimInner))) {
            value = A[batch* batchASize + col * uniforms.dimAOuter + row];
          }
          return value;`;
    }

    let sampleB;
    if (this.transposeB === false) {
      sampleB =
          `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimInner, uniforms.dimBOuter))) {
            value = B[batch * batchBSize + row * uniforms.dimBOuter + col];
          }
          return value;`;
    } else {
      sampleB =
          `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimInner, uniforms.dimBOuter))) {
            value = B[batch * batchBSize + col * uniforms.dimInner + row];
          }
          return value;`;
    }

    // atomicAdd only supports uint/int type. For float, we use
    // atomicCompareExchangeWeak to simulate.
    const atomicAddSnippet = `
     var oldValue = atomicLoad(&(result[flatIndex]));
     var exchanged = false;
     for (; !exchanged;) {
       let newValueF32 = bitcast<f32>(oldValue) + value;
       let newValue = bitcast<i32>(newValueF32);
       let res = atomicCompareExchangeWeak(&(result[flatIndex]), oldValue, newValue);
       oldValue = res.old_value;
       exchanged = res.exchanged;
     }
     `;

    const userCode = `
      fn mm_readA(batch: i32, row : i32, col : i32) -> f32 {
        let batchASize = uniforms.aShape[1] * uniforms.aShape[2];
        var value = 0.0;
        ${sampleA}
      }

      fn mm_readB(batch: i32, row : i32, col : i32) -> f32 {
        let batchBSize = uniforms.bShape[1] * uniforms.bShape[2];
        var value = 0.0;
        ${sampleB}
      }

      fn mm_write(batch: i32, row : i32, col : i32, valueIn : f32) {
        if (row < uniforms.dimAOuter && col < uniforms.dimBOuter) {
          let coords = vec3<i32>(batch, row, col);
          let flatIndex = getOutputIndexFromCoords(coords);
          var value = valueIn;
          // The problem is that we should initialize output to zero before using.
          // Otherwise, the original value will be added to the result.
          ${atomicAddSnippet}
        }
      }

      ${this.makeMatMulSplitKSource()}
    `;
    return userCode;
  }

  makeMatMulSplitKSource(): string {
    const tileAOuter = this.workGroupSize[1] * this.elementsPerThread[1];
    const tileBOuter = this.workGroupSize[0] * this.elementsPerThread[0];
    const rowPerThread = this.elementsPerThread[1];
    const colPerThread = this.elementsPerThread[0];
    const colPerThreadA = this.tileInner / this.workGroupSize[0];
    const rowPerThreadB = this.tileInner / this.workGroupSize[1];
    util.assert(
        this.tileInner % this.workGroupSize[0] === 0 &&
            this.tileInner % this.workGroupSize[1] === 0,
        () =>
            `tileInner ${this.tileInner} must be divisible by workGroupSize[0]${
                this.workGroupSize[0]} and workGroupSize[1]${
                this.workGroupSize[1]}`);
    return `
      var<workgroup> mm_Asub : array<array<f32, ${this.tileInner}>, ${
        tileAOuter}>;
      var<workgroup> mm_Bsub : array<array<f32, ${tileBOuter}>, ${
        this.tileInner}>;
      ${getMainHeaderString()}
        let tileRow = i32(localId.y) * ${rowPerThread};
        let tileCol = i32(localId.x) * ${colPerThread};

        let globalRow = i32(globalId.y) * ${rowPerThread};
        let globalCol = i32(globalId.x) * ${colPerThread};
        let batch = 0;
        let kStart = i32(globalId.z) * ${this.tileInner};

        // Load one tile of A into local memory.
        let tileColA = i32(localId.x) * ${colPerThreadA};
        for (var innerRow = 0; innerRow < ${
        rowPerThread}; innerRow = innerRow + 1) {
          for (var innerCol = 0; innerCol < ${
        colPerThreadA}; innerCol = innerCol + 1) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileColA + innerCol;
            mm_Asub[inputRow][inputCol] = mm_readA(${
        this.batchAEqualOne ? 0 : 'batch'},
                globalRow + innerRow,
                kStart + inputCol);
          }
        }
        // Load one tile of B into local memory.
        let tileRowB = i32(localId.y) * ${rowPerThreadB};
        for (var innerRow = 0; innerRow < ${
        rowPerThreadB}; innerRow = innerRow + 1) {
          for (var innerCol = 0; innerCol < ${
        colPerThread}; innerCol = innerCol + 1) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol + innerCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(${
        this.batchBEqualOne ? 0 : 'batch'},
                kStart + inputRow,
                globalCol + innerCol);
          }
        }

        workgroupBarrier();

        var acc : array<array<f32, ${colPerThread}>, ${rowPerThread}>;
        // Loop over shared dimension. Compute acc values for a single thread.
        for (var k = 0; k < ${this.tileInner}; k = k + 1) {
          var BCached : array<f32, ${colPerThread}>;
          for (var inner = 0; inner < ${colPerThread}; inner = inner + 1) {
            BCached[inner] = mm_Bsub[k][tileCol + inner];
          }

          for (var innerRow = 0; innerRow < ${
        rowPerThread}; innerRow = innerRow + 1) {
            let ACached = mm_Asub[tileRow + innerRow][k];
            for (var innerCol = 0; innerCol < ${
        colPerThread}; innerCol = innerCol + 1) {
              acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];
            }
          }
        }

        for (var innerRow = 0; innerRow < ${
        rowPerThread}; innerRow = innerRow + 1) {
          for (var innerCol = 0; innerCol < ${
        colPerThread}; innerCol = innerCol + 1) {
            mm_write(batch, globalRow + innerRow, globalCol + innerCol, acc[innerRow][innerCol]);
          }
        }
      }
    `;
  }
}

export class BiasActivationProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  uniforms = '';
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  workGroupSize: [number, number, number] = [64, 1, 1];
  size = true;
  private addBias: boolean;
  private activation: backend_util.Activation;
  private hasPreluActivationWeights: boolean;

  constructor(
      outputShape: number[], bias: TensorInfo = null,
      activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.addBias = bias != null;
    this.hasPreluActivationWeights = preluActivationWeights != null;
    this.activation = activation;
    if (this.addBias) {
      this.variableNames.push('bias');
    }

    if (this.hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.shaderKey = `biasActivation_${activation}`;
  }

  getUserCode(): string {
    return `
    ${activationFnSnippet(this.activation, this.hasPreluActivationWeights)}
    ${getMainHeaderAndGlobalIndexString()}
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        var value = getXByOutputIndex(index);
        ${biasActivationSnippet(this.addBias, this.activation)}
        setOutputAtIndex(index, value);
      }
    }
    `;
  }
}
