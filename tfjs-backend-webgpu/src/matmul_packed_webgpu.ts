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

import {backend_util, TensorInfo, util} from '@tensorflow/tfjs-core';
import {activationFnSnippet, biasActivationSnippet, typeSnippet} from './activation_util';
import {getMainHeaderString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, computeWorkGroupSizeForMatMul} from './webgpu_util';

export function matMulReadFnSource(
    batchAEqualOne: boolean, batchBEqualOne: boolean, transposeA: boolean,
    transposeB: boolean, fitAOuter = false, fitBOuter = false, fitInner = false,
    component = 1) {
  util.assert(
      transposeA && component === 1 || !transposeA,
      () => `transposeA ${transposeA} is not compatible with component size ${
          component}`);
  const sampleA = `
      let batch = ${batchAEqualOne ? '0' : 'batchIn'};
      let batchASize = uniforms.aShape[1] * uniforms.aShape[2];
      ${
      transposeA ?
          `value = A[(batch * batchASize + col * uniforms.aShape[2] + row) / ${
              component}];` :
          `value = A[(batch * batchASize + row * uniforms.aShape[2] + col) / ${
              component}];`}

    `;
  let sampleB;
  if (transposeB === false) {
    sampleB =
        `value = B[(batch * batchBSize + row * uniforms.bShape[2] + col) / ${
            component}];`;
  } else {
    sampleB =
        `value = B[(batch * batchBSize + col * uniforms.bShape[2] + row) / ${
            component}];`;
  }

  return `
  fn mm_readA(batchIn: i32, row: i32, colIn: i32) -> ${typeSnippet(component)} {
    var value = ${typeSnippet(component)}(0.0);
    let col = colIn * ${component};
    ${
      fitAOuter && fitInner ?
          sampleA :
          `
    ${
              transposeA ?
                  `if(row < uniforms.dimAOuter && col < uniforms.dimInner)` :
                  `if(row < uniforms.aShape[1] && col < uniforms.aShape[2])`}
    {
      ${sampleA}
    }
    `}
    return value;
  }

  fn mm_readB(batchIn: i32, row: i32, colIn: i32) -> ${typeSnippet(component)} {
    let col = colIn * ${component};
    let batch = ${batchBEqualOne ? '0' : 'batchIn'};
    let batchBSize = uniforms.bShape[1] * uniforms.bShape[2];
    var value = ${typeSnippet(component)}(0.0);
    ${sampleB}
    return value;
  }
  `;
}

export function matMulReadWriteFnSource(
    hasBias: boolean, activation: backend_util.Activation,
    batchAEqualOne: boolean, batchBEqualOne: boolean, transposeA: boolean,
    transposeB: boolean, fitAOuter = false, fitBOuter = false, fitInner = false,
    component = 1) {
  return `
  ${
      matMulReadFnSource(
          batchAEqualOne, batchBEqualOne, transposeA, transposeB, fitAOuter,
          fitBOuter, fitInner, component)}
  fn mm_write(batch: i32, row: i32, colIn: i32, valueIn: ${
      typeSnippet(component)}) {
    let col = colIn * ${component};
    ${
      fitAOuter && fitBOuter ?
          '' :
          'if (row < uniforms.dimAOuter && col < uniforms.dimBOuter)'}
    {
      var value = valueIn;
      let coords = vec3<i32>(batch, row, col);
      ${biasActivationSnippet(hasBias, activation)}
      setOutputAtCoords(coords[0], coords[1], coords[2], value);
    }
  }
  `;
}

const writeDataToSubASnippet = (transpose: boolean) => {
  if (transpose) {
    return `
        mm_Asub[inputRow][inputCol] = mm_readA(batch,
          t * TileInner + inputRow,
          globalRowStart + inputCol);
        `;

  } else {
    return `
        mm_Asub[inputRow][inputCol] = mm_readA(batch,
          globalRowStart + inputRow,
          t * TileInner + inputCol);
        `;
  }
};

const readDataFromSubASnippet = (transposeA: boolean) => {
  return transposeA ? 'let ACached = mm_Asub[k][tileRow + innerRow];' :

                      'let ACached = mm_Asub[tileRow + innerRow][k];';
};

export function makeMatMulPackedSource(
    workPerThread: number[], workGroupSize: [number, number, number],
    transposeA = false, tileInner = 32): string {
  const tileAOuter = workPerThread[1] * workGroupSize[1];
  const tileBOuter = workPerThread[0] * workGroupSize[0];
  const tileAWidth = transposeA ? tileAOuter : tileInner;
  const tileAHight = transposeA ? tileInner : tileAOuter;
  util.assert(
      tileAHight % workGroupSize[1] === 0 &&
          tileAWidth % workGroupSize[0] === 0 &&
          tileInner % workGroupSize[1] === 0,
      () => `tileAHight ${tileAHight} must be divisible by workGroupSize[1]${
          workGroupSize[1]}, tileAWidth ${
          tileAWidth} must be divisible by workGroupSize[0]${
          workGroupSize[0]}, tileInner ${
          tileInner} must be divisible by workGroupSize[1]${workGroupSize[1]}`);
  const rowPerThreadA = tileAHight / workGroupSize[1];
  const colPerThreadA = tileAWidth / workGroupSize[0];
  const rowPerThreadB = tileInner / workGroupSize[1];
  return `
    var<workgroup> mm_Asub : array<array<f32, ${tileAWidth}>, ${tileAHight}>;
    var<workgroup> mm_Bsub : array<array<f32, ${tileBOuter}>, ${tileInner}>;
    const RowPerThread = ${workPerThread[1]};
    const ColPerThread = ${workPerThread[0]};
    const TileInner = ${tileInner};

    @compute @workgroup_size(workGroupSizeX, workGroupSizeY, workGroupSizeZ)
    fn main(@builtin(local_invocation_id) LocalId : vec3<u32>,
            @builtin(global_invocation_id) GlobalId : vec3<u32>,
            @builtin(num_workgroups) NumWorkgroups: vec3<u32>,
            @builtin(workgroup_id) workgroupId: vec3<u32>) {
      localId = LocalId;
      globalId = GlobalId;
      numWorkgroups = NumWorkgroups;

      let tileRow = i32(localId.y) * RowPerThread;
      let tileCol = i32(localId.x) * ColPerThread;

      let globalRow = i32(globalId.y) * RowPerThread;
      let globalCol = i32(globalId.x) * ColPerThread;
      let batch = i32(globalId.z);
      let globalRowStart = i32(workgroupId.y) * ${tileAOuter};

      let numTiles = (uniforms.dimInner - 1) / TileInner + 1;

      var acc : array<array<f32, ColPerThread>, RowPerThread>;

      // Without this initialization strange values show up in acc.
      for (var innerRow = 0; innerRow < RowPerThread; innerRow = innerRow + 1) {
        for (var innerCol = 0; innerCol < ColPerThread; innerCol = innerCol + 1) {
          acc[innerRow][innerCol] = 0.0;
        }
      }

      let tileRowA = i32(localId.y) * ${rowPerThreadA};
      let tileColA = i32(localId.x) * ${colPerThreadA};
      let tileRowB = i32(localId.y) * ${rowPerThreadB};
      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t = t + 1) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < ${
      rowPerThreadA}; innerRow = innerRow + 1) {
          for (var innerCol = 0; innerCol < ${
      colPerThreadA}; innerCol = innerCol + 1) {
            let inputRow = tileRowA + innerRow;
            let inputCol = tileColA + innerCol;
            ${writeDataToSubASnippet(transposeA)}
          }
        }

        // Load one tile of B into local memory.
        for (var innerRow = 0; innerRow < ${
      rowPerThreadB}; innerRow = innerRow + 1) {
          for (var innerCol = 0; innerCol < ColPerThread; innerCol = innerCol + 1) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol + innerCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(batch,
              t * TileInner + inputRow,
              globalCol + innerCol);
          }
        }

        workgroupBarrier();

        // Compute acc values for a single thread.
        var BCached : array<f32, ColPerThread>;
        for (var k = 0; k < TileInner; k = k + 1) {
          for (var inner = 0; inner < ColPerThread; inner = inner + 1) {
            BCached[inner] = mm_Bsub[k][tileCol + inner];
          }

          for (var innerRow = 0; innerRow < RowPerThread; innerRow = innerRow + 1) {
            ${readDataFromSubASnippet(transposeA)}
            for (var innerCol = 0; innerCol < ColPerThread; innerCol = innerCol + 1) {
              acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];
            }
          }
        }

        workgroupBarrier();
      }

      for (var innerRow = 0; innerRow < RowPerThread; innerRow = innerRow + 1) {
        for (var innerCol = 0; innerCol < ColPerThread; innerCol = innerCol + 1) {
          mm_write(batch, globalRow + innerRow, globalCol + innerCol,
              acc[innerRow][innerCol]);
        }
      }
    }
  `;
}

const readVectorASnippet = (transpose: boolean) => {
  return transpose ? `
      mm_readA(batch, colA, globalRow),
      mm_readA(batch, colA + 1, globalRow),
      mm_readA(batch, colA + 2, globalRow),
      mm_readA(batch, colA + 3, globalRow)
  ` :
                     `
      mm_readA(batch, globalRow, colA),
      mm_readA(batch, globalRow, colA + 1),
      mm_readA(batch, globalRow, colA + 2),
      mm_readA(batch, globalRow, colA + 3)
  `;
};

export function makeVectorMatrixProductSource(
    workGroupSize: [number, number, number], transposeA = false): string {
  util.assert(
      workGroupSize[1] === 1 && workGroupSize[2] === 1,
      () => `A linear work group size is required. But got ${workGroupSize}.`);
  return `
    const TileSize = ${workGroupSize[0] * 4};
    var<workgroup> mm_Asub : array<vec4<f32>, ${workGroupSize[0]}>;

    ${getMainHeaderString()}
      let tileCol = i32(localId.x);
      let globalCol = i32(globalId.x);
      let globalRow = i32(globalId.y);

      let numTiles = (uniforms.dimInner - 1) / TileSize + 1;
      let batch = i32(globalId.z);
      // Without this initialization strange values show up in acc.
      var acc = 0.0;

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t = t + 1) {
        // Load one tile of A into local memory.
        let colA = t * TileSize + tileCol * 4;
        mm_Asub[tileCol] = vec4<f32>(${readVectorASnippet(transposeA)});
        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < TileSize / 4; k = k + 1) {
          let rowB = t * TileSize + k * 4;
          let BCached = vec4<f32>(mm_readB(batch, rowB, globalCol),
                              mm_readB(batch, rowB + 1, globalCol),
                              mm_readB(batch, rowB + 2, globalCol),
                              mm_readB(batch, rowB + 3, globalCol));

          let ACached = mm_Asub[k];
          acc = acc + dot(ACached, BCached);
        }

        workgroupBarrier();
      }

      mm_write(batch, globalRow, globalCol, acc);
    }
  `;
}

export class MatMulPackedProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  workPerThread: number;
  variableNames = ['A', 'B'];
  uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workGroupSize: [number, number, number] = [16, 16, 1];
  transposeA: boolean;
  transposeB: boolean;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  batchAEqualOne: boolean;
  batchBEqualOne: boolean;
  fitAOuter: boolean;
  fitBOuter: boolean;
  fitInner: boolean;
  tileInner: number;

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      workPerThread: number, batchAEqualOne: boolean, batchBEqualOne: boolean,
      transposeA = false, transposeB = false, bias: TensorInfo = null,
      activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    const dimInner = transposeA ? aShape[1] : aShape[2];
    this.workGroupSize =
        computeWorkGroupSizeForMatMul(outputShape[1], dimInner, outputShape[2]);
    if (outputShape[1] === 1 || outputShape[2] === 1) {
      workPerThread = 1;
    }
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [workPerThread, workPerThread, 1]);
    // If dispaching number is one, it means only one work group is running.
    // For modern GPUs, it supports multiple work groups running in parallel.
    // So there may be some idle hardware threads.
    // In this case, we prefer to reduce the work per thread and improve the
    // thread utilization
    if (util.arraysEqual(this.dispatch, [1, 1, 1])) {
      workPerThread = 1;
      this.dispatch = computeDispatch(
          this.dispatchLayout, this.outputShape, this.workGroupSize,
          [workPerThread, workPerThread, 1]);
    }
    const addBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.workPerThread = workPerThread;
    this.transposeA = transposeA;
    this.transposeB = transposeB;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    this.batchAEqualOne = batchAEqualOne;
    this.batchBEqualOne = batchBEqualOne;
    [this.fitAOuter, this.fitBOuter, this.fitInner] =
        this.getShapeFit(outputShape[1], outputShape[2], dimInner);
    this.shaderKey = `matMulPacked_${this.workPerThread}_${transposeA}_${
        transposeB}_${this.activation}_${this.fitAOuter}_${this.fitBOuter}_${
        this.fitInner}_${this.outputShape[1] > 1}_${this.batchAEqualOne}_${
        this.batchBEqualOne}`;
  }

  getShapeFit(dimAOuter: number, dimBOuter: number, dimInner: number):
      boolean[] {
    const tileAOuter = this.workGroupSize[1] * this.workPerThread;
    const tileBOuter = this.workGroupSize[0] * this.workPerThread;
    this.tileInner = 32;

    if (this.outputShape[1] === 1) {
      this.tileInner = this.workGroupSize[0] * 4;
    }

    const fitAOuter = dimAOuter % tileAOuter === 0;
    const fitBOuter = dimBOuter % tileBOuter === 0;
    const fitInner = dimInner % this.tileInner === 0;
    return [fitAOuter, fitBOuter, fitInner];
  }

  getUserCode(): string {
    const userCode = `
      ${activationFnSnippet(this.activation, this.hasPreluActivationWeights)}
      ${
        matMulReadWriteFnSource(
            this.addBias, this.activation, this.batchAEqualOne,
            this.batchBEqualOne,
            false /* transposeA is implemented in makeMatMulPackedSource */,
            this.transposeB, this.fitAOuter, this.fitBOuter, this.fitInner)}
      ${
        this.outputShape[1] > 1 ?
            makeMatMulPackedSource(
                [this.workPerThread, this.workPerThread, 1], this.workGroupSize,
                this.transposeA, this.tileInner) :
            makeVectorMatrixProductSource(this.workGroupSize, this.transposeA)}
    `;
    return userCode;
  }
}
