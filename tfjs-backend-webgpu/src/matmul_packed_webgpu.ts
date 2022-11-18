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
import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, computeWorkgroupInfoForMatMul} from './webgpu_util';

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
      ${
      transposeA ? `value = getA(batch, col, row);` :
                   `value = getA(batch, row, col);`}

    `;
  const sampleB = transposeB ? `value = getB(batch, col, row);` :
                               `value = getB(batch, row, col);`;

  return `
  fn mm_readA(batchIn: i32, row: i32, colIn: i32) -> ${
      typeSnippet(component)} {
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

  fn mm_readB(batchIn: i32, row: i32, colIn: i32) -> ${
      typeSnippet(component)} {
    let col = colIn * ${component};
    let batch = ${batchBEqualOne ? '0' : 'batchIn'};
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

const writeDataToSubAVec4Snippet = (transpose: boolean) => {
  if (transpose) {
    return `
        mm_Asub[inputRow][inputCol] = mm_readA(batch,
          kStart + inputRow,
          globalRowStart / innerElementSize + inputCol);
        `;

  } else {
    return `
        mm_Asub[inputRow][inputCol] = mm_readA(batch,
          globalRow + innerRow,
          kStart / innerElementSize + inputCol);
        `;
  }
};

const calculateResultSnippet =
    (transposeA: boolean, innerElementSize: number) => {
      if (transposeA) {
        return `
        let ACached0 = mm_Asub[k * innerElementSize][localRow];
        let ACached1 = mm_Asub[k * innerElementSize + 1][localRow];
        let ACached2 = mm_Asub[k * innerElementSize + 2][localRow];
        ${
            innerElementSize === 3 ?
                '' :
                'let ACached3 = mm_Asub[k * innerElementSize + 3][localRow];'}
        for (var i = 0; i < rowPerThread; i = i + 1) {
          acc[i] = BCached0 * ACached0[i] + acc[i];
          acc[i] = BCached1 * ACached1[i] + acc[i];
          acc[i] = BCached2 * ACached2[i] + acc[i];
          ${
            innerElementSize === 3 ?
                '' :
                'acc[i] = BCached3 * ACached3[i] + acc[i];'}
        }`;
      } else {
        return `
        for (var i = 0; i < rowPerThread; i = i + 1) {
          let ACached = mm_Asub[tileRow + i][k];
          acc[i] = BCached0 * ACached.x + acc[i];
          acc[i] = BCached1 * ACached.y + acc[i];
          acc[i] = BCached2 * ACached.z + acc[i];
          ${
            innerElementSize === 3 ? '' :
                                     'acc[i] = BCached3 * ACached.w + acc[i];'}
        }`;
      }
    };

export function makeMatMulPackedVec4Source(
    workPerThread: number[], workgroupSize: [number, number, number],
    transposeA = false, tileInner = 32, splitK = false, splitedDimInner = 32,
    isVectorA = false): string {
  const tileAOuter = workgroupSize[1] * workPerThread[1];
  const tileBOuter = workgroupSize[0] * workPerThread[0];
  const tileAWidth = transposeA ? tileAOuter : tileInner;
  const tileAHight = transposeA ? tileInner : tileAOuter;
  const innerElementSize = tileAWidth / workgroupSize[0];
  const rowPerThreadB = tileInner / workgroupSize[1];
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

  const rowPerThread = ${workPerThread[1]};
  const colPerThread = ${workPerThread[0]};
  const innerElementSize = ${innerElementSize};
  const tileInner = ${tileInner};

  ${main()} {
    let localRow = i32(localId.y);
    let tileRow = ${isVectorA ? '0' : 'localRow * rowPerThread'};
    let tileCol = i32(localId.x);

    let globalRow = ${isVectorA ? '0' : 'i32(globalId.y) * rowPerThread'};
    let globalCol = i32(globalId.x);
    let batch = ${splitK ? '0' : 'i32(globalId.z)'};
    let globalRowStart = i32(workgroupId.y) * ${tileAOuter};

    let numTiles = ${
      splitK ? `${Math.ceil(splitedDimInner / tileInner)}` :
               '(uniforms.dimInner - 1) / tileInner + 1'};
    var kStart = ${splitK ? `i32(globalId.z) * ${splitedDimInner}` : '0'};

    var acc: array<vec4<f32>, rowPerThread>;

    // Loop over shared dimension.
    let tileRowB = localRow * ${rowPerThreadB};
    for (var t = 0; t < numTiles; t = t + 1) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileCol;
            ${writeDataToSubAVec4Snippet(transposeA)}
        }

        // Load one tile of B into local memory.
        for (var innerRow = 0; innerRow < ${
      rowPerThreadB}; innerRow = innerRow + 1) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(batch, kStart + inputRow, globalCol);
        }
        kStart = kStart + tileInner;
        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < tileInner / innerElementSize; k = k + 1) {
            let BCached0 = mm_Bsub[k * innerElementSize][tileCol];
            let BCached1 = mm_Bsub[k * innerElementSize + 1][tileCol];
            let BCached2 = mm_Bsub[k * innerElementSize + 2][tileCol];
            ${
      innerElementSize === 3 ?
          '' :
          'let BCached3 = mm_Bsub[k * innerElementSize + 3][tileCol];'}

            ${calculateResultSnippet(transposeA, innerElementSize)}
        }

        workgroupBarrier();
    }

    for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
        mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);
    }
  }`;
}

const writeDataToSubASnippet = (transpose: boolean) => {
  if (transpose) {
    return `
        mm_Asub[inputRow][inputCol] = mm_readA(batch,
          kStart + inputRow,
          globalRowStart + inputCol);
        `;

  } else {
    return `
        mm_Asub[inputRow][inputCol] = mm_readA(batch,
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
    sequentialAccessByThreads = false): string {
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
  const matmulSnippet = sequentialAccessByThreads ?
      `
      let localRow = i32(localId.y);
      let localCol = i32(localId.x);
      let globalRowStart = i32(workgroupId.y) * ${tileAOuter};
      let globalColStart = i32(workgroupId.x) * ${tileBOuter};

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t = t + 1) {
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
            mm_Bsub[inputRow][inputCol] = mm_readB(batch,
              kStart + inputRow,
              globalColStart + inputCol);
          }
        }
        kStart = kStart + tileInner;
        workgroupBarrier();

        // Compute acc values for a single thread.
        var BCached : array<f32, colPerThread>;
        for (var k = 0; k < tileInner; k = k + 1) {
          for (var inner = 0; inner < colPerThread; inner = inner + 1) {
            BCached[inner] = mm_Bsub[k][localCol + inner * ${
          workgroupSize[0]}];
          }
          for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
            let ACached = ${
          transposeA ?
              `mm_Asub[k][localRow + innerRow * ${workgroupSize[1]}];` :
              `mm_Asub[localRow + innerRow * ${workgroupSize[1]}][k];`}
            for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
              acc[innerRow][innerCol] = acc[innerRow][innerCol] +
                  ACached * BCached[innerCol];
            }
          }
        }
        workgroupBarrier();
      }
      for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
        let gRow = globalRowStart + localRow + innerRow * ${workgroupSize[1]};
        for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
          let gCol = globalColStart + localCol + innerCol * ${
          workgroupSize[0]};
          mm_write(batch, gRow, gCol, acc[innerRow][innerCol]);
        }
      }
      ` :
      `
  let tileRow = i32(localId.y) * rowPerThread;
  let tileCol = i32(localId.x) * colPerThread;

  let globalRow = i32(globalId.y) * rowPerThread;
  let globalCol = i32(globalId.x) * colPerThread;
  let globalRowStart = i32(workgroupId.y) * ${tileAOuter};

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
      for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
        let inputRow = tileRowB + innerRow;
        let inputCol = tileCol + innerCol;
        mm_Bsub[inputRow][inputCol] = mm_readB(batch,
          kStart + inputRow,
          globalCol + innerCol);
      }
    }
    kStart = kStart + tileInner;
    workgroupBarrier();

    // Compute acc values for a single thread.
    var BCached : array<f32, colPerThread>;
    for (var k = 0; k < tileInner; k = k + 1) {
      for (var inner = 0; inner < colPerThread; inner = inner + 1) {
        BCached[inner] = mm_Bsub[k][tileCol + inner];
      }

      for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
        ${readDataFromSubASnippet(transposeA)}
        for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
          acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];
        }
      }
    }

    workgroupBarrier();
  }

  for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
    for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
      mm_write(batch, globalRow + innerRow, globalCol + innerCol,
          acc[innerRow][innerCol]);
    }
  }
  `;

  return `
    var<workgroup> mm_Asub : array<array<f32, ${tileAWidth}>, ${tileAHight}>;
    var<workgroup> mm_Bsub : array<array<f32, ${tileBOuter}>, ${tileInner}>;
    const rowPerThread = ${workPerThread[1]};
    const colPerThread = ${workPerThread[0]};
    const tileInner = ${tileInner};

    ${main()} {
      let batch = ${splitK ? '0' : 'i32(globalId.z)'};
      let numTiles = ${
      splitK ? `${Math.ceil(splitedDimInner / tileInner)}` :
               '(uniforms.dimInner - 1) / tileInner + 1'};
      var kStart = ${splitK ? `i32(globalId.z) * ${splitedDimInner}` : '0'};

      var acc : array<array<f32, colPerThread>, rowPerThread>;

      // Without this initialization strange values show up in acc.
      for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {
        for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {
          acc[innerRow][innerCol] = 0.0;
        }
      }
      ${matmulSnippet}
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
    workgroupSize: [number, number, number], transposeA = false): string {
  util.assert(
      workgroupSize[1] === 1 && workgroupSize[2] === 1,
      () => `A linear work group size is required. But got ${workgroupSize}.`);
  return `
    const tileSize = ${workgroupSize[0] * 4};
    var<workgroup> mm_Asub : array<vec4<f32>, ${workgroupSize[0]}>;

    ${main()} {
      let tileCol = i32(localId.x);
      let globalCol = i32(globalId.x);
      let globalRow = i32(globalId.y);

      let numTiles = (uniforms.dimInner - 1) / tileSize + 1;
      let batch = i32(globalId.z);
      // Without this initialization strange values show up in acc.
      var acc = 0.0;

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t = t + 1) {
        // Load one tile of A into local memory.
        let colA = t * tileSize + tileCol * 4;
        mm_Asub[tileCol] = vec4<f32>(${readVectorASnippet(transposeA)});
        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < tileSize / 4; k = k + 1) {
          let rowB = t * tileSize + k * 4;
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
  variableNames = ['A', 'B'];
  uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workgroupSize: [number, number, number];
  elementsPerThread: [number, number, number];
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
  isVectorA: boolean;
  isVec4: boolean;
  private sequentialAccessByThreads: boolean;

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      batchAEqualOne: boolean, batchBEqualOne: boolean, transposeA = false,
      transposeB = false, bias: TensorInfo = null,
      activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null,
      sequentialAccessByThreads = false) {
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    const dimInner = transposeA ? aShape[1] : aShape[2];
    this.isVec4 = ((dimInner % 4 === 0 && !transposeA) ||
                   (outputShape[1] % 4 === 0 && transposeA)) &&
        outputShape[2] % 4 === 0 && !transposeB;
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
    this.batchAEqualOne = batchAEqualOne;
    this.batchBEqualOne = batchBEqualOne;
    [this.fitAOuter, this.fitBOuter, this.fitInner] =
        this.getShapeFit(outputShape[1], outputShape[2], dimInner);
    this.shaderKey = `matMulPacked_${this.elementsPerThread}_${transposeA}_${
        transposeB}_${this.activation}_${this.fitAOuter}_${this.fitBOuter}_${
        this.fitInner}_${this.isVec4}_${this.isVectorA}_${
        this.batchAEqualOne}_${this.batchBEqualOne}_${
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
            this.addBias, this.activation, this.batchAEqualOne,
            this.batchBEqualOne,
            false /* transposeA is implemented in makeMatMulPackedSource */,
            this.transposeB, this.fitAOuter, this.fitBOuter, this.fitInner,
            this.isVec4 ? 4 : 1)}
      ${
        this.isVec4 ?
            makeMatMulPackedVec4Source(
                this.elementsPerThread, this.workgroupSize, this.transposeA,
                this.tileInner, false, null, this.isVectorA) :
            (this.isVectorA ? makeVectorMatrixProductSource(
                                  this.workgroupSize, this.transposeA) :
                              makeMatMulPackedSource(
                                  this.elementsPerThread, this.workgroupSize,
                                  this.transposeA, this.tileInner, false, null,
                                  this.sequentialAccessByThreads))}
    `;
    return userCode;
  }
}
