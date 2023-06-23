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

import {activationFnSnippet, biasActivationSnippet} from './activation_util';
import {getMainHeaderString as main, typeSnippet, WebGPUProgram} from './webgpu_program';
import {computeDispatch, computeWorkgroupInfoForMatMul} from './webgpu_util';

export function matMulReadFnSource(
    transposeA: boolean, transposeB: boolean, fitAOuter = false,
    fitBOuter = false, fitInner = false, component = 1) {
  util.assert(
      transposeA && component === 1 || !transposeA,
      () => `transposeA ${transposeA} is not compatible with component size ${
          component}`);
  const sampleA = `
      ${
      transposeA ? `value = getA(batch, col, row);` :
                   `value = getA(batch, row, col);`}

    `;
  const sampleB = transposeB ? `value = getB(batch, col, row);` :
                               `value = getB(batch, row, col);`;

  return `
  fn mm_readA(batch: i32, row: i32, col: i32) -> ${typeSnippet(component)} {
    var value = ${typeSnippet(component)}(0.0);
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

  fn mm_readB(batch: i32, row: i32, col: i32) -> ${typeSnippet(component)} {
    var value = ${typeSnippet(component)}(0.0);
    ${sampleB}
    return value;
  }
  `;
}

export function matMulReadWriteFnSource(
    hasBias: boolean, activation: backend_util.Activation, transposeA: boolean,
    transposeB: boolean, fitAOuter = false, fitBOuter = false, fitInner = false,
    component = 1) {
  return `
  ${
      matMulReadFnSource(
          transposeA, transposeB, fitAOuter, fitBOuter, fitInner, component)}
  fn mm_write(batch: i32, row: i32, col: i32, valueIn: ${
      typeSnippet(component)}) {
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

const writeDataToSubAVec4Snippet =
    (transpose: boolean, innerElementSize: number) => {
      if (transpose) {
        return `
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          kStart + inputRow,
          globalRowStart + inputCol * ${innerElementSize});
        `;

      } else {
        return `
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          globalRow + innerRow,
          kStart + inputCol * ${innerElementSize});
        `;
      }
    };

const calculateResultSnippet =
    (transposeA: boolean, innerElementSize: number, rowPerThread: number,
     tileInner: number) => {
      if (transposeA) {
        return `
      for (var k = 0; k < ${tileInner}; k++) {
        let BCached0 = mm_Bsub[k][tileCol];
        let ACached0 = mm_Asub[k][localRow];
        for (var i = 0; i < ${rowPerThread}; i++) {
          acc[i] = fma(BCached0, vec4<f32>(ACached0[i]), acc[i]);
        }
      }`;
      } else {
        let bCachedStr = '';
        let accStr = '';
        for (let i = 0; i < innerElementSize; i++) {
          bCachedStr += `let BCached${i} = mm_Bsub[k * ${innerElementSize} + ${
              i}][tileCol];`;
          accStr +=
              `acc[i] = fma(BCached${i}, vec4<f32>(ACached[${i}]), acc[i]);`;
        }
        return `
      for (var k = 0; k < ${tileInner / innerElementSize}; k++) {
        ${bCachedStr}
        for (var i = 0; i < ${rowPerThread}; i++) {
          let ACached = mm_Asub[tileRow + i][k];
          ${accStr}
        }
      }`;
      }
    };

export function makeMatMulPackedVec4Source(
    workPerThread: number[], workgroupSize: [number, number, number],
    transposeA = false, tileInner = 32, splitK = false, splitedDimInner = 32,
    broadcastBatch = false): string {
  const tileAOuter = workgroupSize[1] * workPerThread[1];
  const tileBOuter = workgroupSize[0] * workPerThread[0];
  const tileAWidth = transposeA ? tileAOuter : tileInner;
  const tileAHight = transposeA ? tileInner : tileAOuter;
  const innerElementSize = tileAWidth / workgroupSize[0];
  const rowPerThreadB = tileInner / workgroupSize[1];
  const rowPerThread = workPerThread[1];
  const colPerThread = workPerThread[0];
  util.assert(
      ((transposeA && innerElementSize === 4 && workPerThread[1] === 4) ||
       (!transposeA && (innerElementSize === 3 || innerElementSize === 4))) &&
          tileAWidth % workgroupSize[0] === 0 &&
          tileInner % workgroupSize[1] === 0 && workPerThread[0] === 4,
      () => `If transposeA ${transposeA} is true, innerElementSize ${
          innerElementSize} and workPerThread[1] ${workPerThread[1]} must be 4.
          Otherwise, innerElementSize ${innerElementSize} must be 3 or 4.
      tileAWidth ${tileAWidth} must be divisible by workgroupSize[0]${
          workgroupSize[0]}. tileInner ${
          tileInner} must be divisible by workgroupSize[1] ${
          workgroupSize[1]}. colPerThread ${workPerThread[0]} must be 4.`);
  return `
  var<workgroup> mm_Asub : array<array<vec${innerElementSize}<f32>, ${
      tileAWidth / innerElementSize}>, ${tileAHight}>;
  var<workgroup> mm_Bsub : array<array<vec4<f32>, ${
      tileBOuter / workPerThread[0]}>, ${tileInner}>;

  ${main()} {
    let localRow = i32(localId.y);
    let tileRow = localRow * ${rowPerThread};
    let tileCol = i32(localId.x);

    let globalRow = i32(globalId.y) * ${rowPerThread};
    let globalCol = i32(globalId.x) * ${colPerThread};
    let batch = ${splitK ? '0' : 'i32(globalId.z)'};
    let batchA = ${
      splitK || !broadcastBatch ? 'batch' : 'batch % uniforms.aShape[0]'};
    let batchB = ${
      splitK || !broadcastBatch ? 'batch' : 'batch % uniforms.bShape[0]'};
    let globalRowStart = i32(workgroupId.y) * ${tileAOuter};

    let numTiles = ${
      splitK ? `${Math.ceil(splitedDimInner / tileInner)}` :
               `(uniforms.dimInner - 1) / ${tileInner} + 1`};
    var kStart = ${splitK ? `i32(globalId.z) * ${splitedDimInner}` : '0'};

    var acc: array<vec4<f32>, ${rowPerThread}>;

    // Loop over shared dimension.
    let tileRowB = localRow * ${rowPerThreadB};
    for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < ${rowPerThread}; innerRow++) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileCol;
            ${writeDataToSubAVec4Snippet(transposeA, innerElementSize)}
        }

        // Load one tile of B into local memory.
        for (var innerRow = 0; innerRow < ${rowPerThreadB}; innerRow++) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(batchB, kStart + inputRow, globalCol);
        }
        kStart = kStart + ${tileInner};
        workgroupBarrier();

        // Compute acc values for a single thread.
        ${
      calculateResultSnippet(
          transposeA, innerElementSize, rowPerThread, tileInner)}
        workgroupBarrier();
    }

    for (var innerRow = 0; innerRow < ${rowPerThread}; innerRow++) {
        mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);
    }
  }`;
}

const writeDataToSubASnippet = (transpose: boolean) => {
  if (transpose) {
    return `
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          kStart + inputRow,
          globalRowStart + inputCol);
        `;

  } else {
    return `
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          globalRowStart + inputRow,
          kStart + inputCol);
        `;
  }
};

const readDataFromSubASnippet = (transposeA: boolean) => {
  return transposeA ? 'let ACached = mm_Asub[k][tileRow + innerRow];' :

                      'let ACached = mm_Asub[tileRow + innerRow][k];';
};

// sequentialAccessByThreads means sequential data in memory is accessed by
// threads, instead of a single thread (default behavior).
export function makeMatMulPackedSource(
    workPerThread: number[], workgroupSize: [number, number, number],
    transposeA = false, tileInner = 32, splitK = false, splitedDimInner = 32,
    sequentialAccessByThreads = false, broadcastBatch = false): string {
  const tileAOuter = workPerThread[1] * workgroupSize[1];
  const tileBOuter = workPerThread[0] * workgroupSize[0];
  const tileAWidth = transposeA ? tileAOuter : tileInner;
  const tileAHight = transposeA ? tileInner : tileAOuter;
  util.assert(
      tileAHight % workgroupSize[1] === 0 &&
          tileAWidth % workgroupSize[0] === 0 &&
          tileInner % workgroupSize[1] === 0,
      () => `tileAHight ${tileAHight} must be divisible by workgroupSize[1]${
          workgroupSize[1]}, tileAWidth ${
          tileAWidth} must be divisible by workgroupSize[0]${
          workgroupSize[0]}, tileInner ${
          tileInner} must be divisible by workgroupSize[1]${workgroupSize[1]}`);
  const rowPerThreadA = tileAHight / workgroupSize[1];
  const colPerThreadA = tileAWidth / workgroupSize[0];
  const rowPerThreadB = tileInner / workgroupSize[1];
  const rowPerThread = workPerThread[1];
  const colPerThread = workPerThread[0];
  const matmulSnippet = sequentialAccessByThreads ?
      `
      let localRow = i32(localId.y);
      let localCol = i32(localId.x);
      let globalRowStart = i32(workgroupId.y) * ${tileAOuter};
      let globalColStart = i32(workgroupId.x) * ${tileBOuter};

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var inputRow = localRow; inputRow < ${
          tileAHight}; inputRow = inputRow + ${workgroupSize[1]}) {
          for (var inputCol = localCol; inputCol < ${
          tileAWidth}; inputCol = inputCol + ${workgroupSize[0]}) {
            ${writeDataToSubASnippet(transposeA)}
          }
        }
        // Load one tile of B into local memory.
        for (var inputRow = localRow; inputRow < ${
          tileInner}; inputRow = inputRow + ${workgroupSize[1]}) {
              for (var inputCol = localCol; inputCol < ${
          tileBOuter}; inputCol = inputCol + ${workgroupSize[0]}) {
            mm_Bsub[inputRow][inputCol] = mm_readB(batchB,
              kStart + inputRow,
              globalColStart + inputCol);
          }
        }
        kStart = kStart + ${tileInner};
        workgroupBarrier();

        // Compute acc values for a single thread.
        var BCached : array<f32, ${colPerThread}>;
        for (var k = 0; k < ${tileInner}; k++) {
          for (var inner = 0; inner < ${colPerThread}; inner++) {
            BCached[inner] = mm_Bsub[k][localCol + inner * ${workgroupSize[0]}];
          }
          for (var innerRow = 0; innerRow < ${rowPerThread}; innerRow++) {
            let ACached = ${
          transposeA ?
              `mm_Asub[k][localRow + innerRow * ${workgroupSize[1]}];` :
              `mm_Asub[localRow + innerRow * ${workgroupSize[1]}][k];`}
            for (var innerCol = 0; innerCol < ${colPerThread}; innerCol++) {
              acc[innerRow][innerCol] =
                  fma(ACached, BCached[innerCol], acc[innerRow][innerCol]);
            }
          }
        }
        workgroupBarrier();
      }
      for (var innerRow = 0; innerRow < ${rowPerThread}; innerRow++) {
        let gRow = globalRowStart + localRow + innerRow * ${workgroupSize[1]};
        for (var innerCol = 0; innerCol < ${colPerThread}; innerCol++) {
          let gCol = globalColStart + localCol + innerCol * ${workgroupSize[0]};
          mm_write(batch, gRow, gCol, acc[innerRow][innerCol]);
        }
      }
      ` :
      `
  let tileRow = i32(localId.y) * ${rowPerThread};
  let tileCol = i32(localId.x) * ${colPerThread};

  let globalRow = i32(globalId.y) * ${rowPerThread};
  let globalCol = i32(globalId.x) * ${colPerThread};
  let globalRowStart = i32(workgroupId.y) * ${tileAOuter};

  let tileRowA = i32(localId.y) * ${rowPerThreadA};
  let tileColA = i32(localId.x) * ${colPerThreadA};
  let tileRowB = i32(localId.y) * ${rowPerThreadB};
  // Loop over shared dimension.
  for (var t = 0; t < numTiles; t++) {
    // Load one tile of A into local memory.
    for (var innerRow = 0; innerRow < ${rowPerThreadA}; innerRow++) {
      for (var innerCol = 0; innerCol < ${colPerThreadA}; innerCol++) {
        let inputRow = tileRowA + innerRow;
        let inputCol = tileColA + innerCol;
        ${writeDataToSubASnippet(transposeA)}
      }
    }

    // Load one tile of B into local memory.
    for (var innerRow = 0; innerRow < ${rowPerThreadB}; innerRow++) {
      for (var innerCol = 0; innerCol < ${colPerThread}; innerCol++) {
        let inputRow = tileRowB + innerRow;
        let inputCol = tileCol + innerCol;
        mm_Bsub[inputRow][inputCol] = mm_readB(batchB,
          kStart + inputRow,
          globalCol + innerCol);
      }
    }
    kStart = kStart + ${tileInner};
    workgroupBarrier();

    // Compute acc values for a single thread.
    var BCached : array<f32, ${colPerThread}>;
    for (var k = 0; k < ${tileInner}; k++) {
      for (var inner = 0; inner < ${colPerThread}; inner++) {
        BCached[inner] = mm_Bsub[k][tileCol + inner];
      }

      for (var innerRow = 0; innerRow < ${rowPerThread}; innerRow++) {
        ${readDataFromSubASnippet(transposeA)}
        for (var innerCol = 0; innerCol < ${colPerThread}; innerCol++) {
          acc[innerRow][innerCol] =
              fma(ACached, BCached[innerCol], acc[innerRow][innerCol]);
        }
      }
    }

    workgroupBarrier();
  }

  for (var innerRow = 0; innerRow < ${rowPerThread}; innerRow++) {
    for (var innerCol = 0; innerCol < ${colPerThread}; innerCol++) {
      mm_write(batch, globalRow + innerRow, globalCol + innerCol,
          acc[innerRow][innerCol]);
    }
  }
  `;

  return `
    var<workgroup> mm_Asub : array<array<f32, ${tileAWidth}>, ${tileAHight}>;
    var<workgroup> mm_Bsub : array<array<f32, ${tileBOuter}>, ${tileInner}>;

    ${main()} {
      let batch = ${splitK ? '0' : 'i32(globalId.z)'};
      let batchA = ${
      splitK || !broadcastBatch ? 'batch' : 'batch % uniforms.aShape[0]'};
      let batchB = ${
      splitK || !broadcastBatch ? 'batch' : 'batch % uniforms.bShape[0]'};
      let numTiles = ${
      splitK ? `${Math.ceil(splitedDimInner / tileInner)}` :
               `(uniforms.dimInner - 1) / ${tileInner} + 1`};
      var kStart = ${splitK ? `i32(globalId.z) * ${splitedDimInner}` : '0'};

      var acc : array<array<f32, ${colPerThread}>, ${rowPerThread}>;

      // Without this initialization strange values show up in acc.
      for (var innerRow = 0; innerRow < ${rowPerThread}; innerRow++) {
        for (var innerCol = 0; innerCol < ${colPerThread}; innerCol++) {
          acc[innerRow][innerCol] = 0.0;
        }
      }
      ${matmulSnippet}
    }
  `;
}

const readVectorASnippet = (transpose: boolean) => {
  return transpose ? `
      mm_readA(batchA, colA, globalRow),
      mm_readA(batchA, colA + 1, globalRow),
      mm_readA(batchA, colA + 2, globalRow),
      mm_readA(batchA, colA + 3, globalRow)
  ` :
                     `
      mm_readA(batchA, globalRow, colA),
      mm_readA(batchA, globalRow, colA + 1),
      mm_readA(batchA, globalRow, colA + 2),
      mm_readA(batchA, globalRow, colA + 3)
  `;
};

export function makeVectorMatrixProductSource(
    workgroupSize: [number, number, number], transposeA = false): string {
  util.assert(
      workgroupSize[1] === 1 && workgroupSize[2] === 1,
      () => `A linear work group size is required. But got ${workgroupSize}.`);
  const tileSize = workgroupSize[0] * 4;
  return `
    var<workgroup> mm_Asub : array<vec4<f32>, ${workgroupSize[0]}>;

    ${main()} {
      let tileCol = i32(localId.x);
      let globalCol = i32(globalId.x);
      let globalRow = i32(globalId.y);

      let numTiles = (uniforms.dimInner - 1) / ${tileSize} + 1;
      let batch = i32(globalId.z);
      let batchA = batch % uniforms.aShape[0];
      let batchB = batch % uniforms.bShape[0];
      // Without this initialization strange values show up in acc.
      var acc = 0.0;

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        let colA = t * ${tileSize} + tileCol * 4;
        mm_Asub[tileCol] = vec4<f32>(${readVectorASnippet(transposeA)});
        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < ${tileSize / 4}; k++) {
          let rowB = t * ${tileSize} + k * 4;
          let BCached = vec4<f32>(mm_readB(batchB, rowB, globalCol),
                              mm_readB(batchB, rowB + 1, globalCol),
                              mm_readB(batchB, rowB + 2, globalCol),
                              mm_readB(batchB, rowB + 3, globalCol));

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
  variableNames = ['A', 'B'];
  uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workgroupSize: [number, number, number];
  elementsPerThread: [number, number, number];
  transposeA: boolean;
  transposeB: boolean;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  fitAOuter: boolean;
  fitBOuter: boolean;
  fitInner: boolean;
  tileInner: number;
  isVectorA: boolean;
  isVec4: boolean;
  outputComponent: number;
  private sequentialAccessByThreads: boolean;

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      transposeA = false, transposeB = false, bias: TensorInfo = null,
      activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null,
      sequentialAccessByThreads = false) {
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    const dimInner = transposeA ? aShape[1] : aShape[2];
    this.isVec4 = ((dimInner % 4 === 0 && !transposeA) ||
                   (outputShape[1] % 4 === 0 && transposeA)) &&
        outputShape[2] % 4 === 0 && !transposeB;
    this.outputComponent = this.isVec4 ? 4 : 1;
    this.isVectorA = outputShape[1] === 1 && !transposeA;

    if (!this.isVec4 && this.isVectorA) {
      // For makeVectorMatrixProductSource
      this.elementsPerThread = [1, 1, 1];
      this.workgroupSize = [32, 1, 1];
    } else {
      const workgroupInfo = computeWorkgroupInfoForMatMul(
          outputShape[1], dimInner, outputShape[2], transposeA);
      this.workgroupSize = workgroupInfo.workgroupSize;
      this.elementsPerThread = workgroupInfo.elementsPerThread;
    }

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize,
        this.elementsPerThread);

    const addBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.sequentialAccessByThreads = sequentialAccessByThreads;
    this.transposeA = transposeA;
    this.transposeB = transposeB;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    [this.fitAOuter, this.fitBOuter, this.fitInner] =
        this.getShapeFit(outputShape[1], outputShape[2], dimInner);
    this.shaderKey = `matMulPacked_${this.elementsPerThread}_${transposeA}_${
        transposeB}_${this.activation}_${this.fitAOuter}_${this.fitBOuter}_${
        this.fitInner}_${this.isVec4}_${this.isVectorA}_${
        this.sequentialAccessByThreads}`;
  }

  getShapeFit(dimAOuter: number, dimBOuter: number, dimInner: number):
      boolean[] {
    const tileAOuter = this.workgroupSize[1] * this.elementsPerThread[1];
    const tileBOuter = this.workgroupSize[0] * this.elementsPerThread[0];

    if (!this.isVec4 && this.isVectorA) {
      // For makeVectorMatrixProductSource
      this.tileInner = this.workgroupSize[0] * 4;
    } else {
      this.tileInner = tileBOuter;
    }

    const fitAOuter = dimAOuter % tileAOuter === 0;
    const fitBOuter = dimBOuter % tileBOuter === 0;
    const fitInner = dimInner % this.tileInner === 0;
    return [fitAOuter, fitBOuter, fitInner];
  }

  getUserCode(): string {
    const userCode = `
      ${
        activationFnSnippet(
            this.activation, this.hasPreluActivationWeights, this.isVec4)}
      ${
        matMulReadWriteFnSource(
            this.addBias, this.activation,
            false /* transposeA is implemented in makeMatMulPackedSource */,
            this.transposeB, this.fitAOuter, this.fitBOuter, this.fitInner,
            this.isVec4 ? 4 : 1)}
      ${
        this.isVec4 ?
            makeMatMulPackedVec4Source(
                this.elementsPerThread, this.workgroupSize, this.transposeA,
                this.tileInner, false, null, true) :
            (this.isVectorA ? makeVectorMatrixProductSource(
                                  this.workgroupSize, this.transposeA) :
                              makeMatMulPackedSource(
                                  this.elementsPerThread, this.workgroupSize,
                                  this.transposeA, this.tileInner, false, null,
                                  this.sequentialAccessByThreads, true))}
    `;
    return userCode;
  }
}
