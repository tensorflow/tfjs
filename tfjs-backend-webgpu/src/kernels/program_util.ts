/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {util} from '@tensorflow/tfjs-core';
import {Conv2DMMVec4Program} from './conv2d_mm_vec4_webgpu';

import {Conv2DMMProgram} from './conv2d_mm_webgpu';
import {MatMulPackedVec4Program} from './matmul_packed_vec4_webgpu';
import {MatMulPackedProgram} from './matmul_packed_webgpu';

function tilesFitEvenlyIntoShape(tileSize: number[], shape: number[]): boolean {
  if (tileSize.length !== shape.length) {
    throw new Error(
        `Cannot compute whether rank ${tileSize.length}` +
        ` tiles fit evenly into rank ${shape.length} shape` +
        ` - ranks must match.`);
  }
  return shape.every(
      (dim: number, dimIdx: number) => dim % tileSize[dimIdx] === 0);
}

export function getShapeFitForMatMulPackedProgram(program: MatMulPackedProgram):
    boolean[] {
  const tileAOuter = program.workGroupSize[1] * program.workPerThread;
  const tileBOuter = program.workGroupSize[0] * program.workPerThread;
  let tileInner = tileAOuter > tileBOuter ? tileAOuter : tileBOuter;
  if (program.outputShape[1] === 1) {
    tileInner *= 4;
  }
  util.assert(
      tileInner % program.workGroupSize[0] === 0 &&
          tileInner % program.workGroupSize[1] === 0,
      () => `tileInner must be multiple of workgroupsize.x ` +
          `and workgroupsize.y`);
  const tileSizeA = [tileAOuter, tileInner];
  const tileSizeB = [tileInner, tileBOuter];

  return [
    tilesFitEvenlyIntoShape(tileSizeA, program.aShape.slice(1)),
    tilesFitEvenlyIntoShape(tileSizeB, program.bShape.slice(1))
  ];
}

export function getShapeFitForMatMulPackedVec4Program(
    program: MatMulPackedVec4Program): boolean[] {
  const dimInner = program.aShape[2];
  const dimBOuter = program.outputShape[2];
  const bShape = [program.outputShape[0], dimInner, dimBOuter];
  const tileAOuter = program.workGroupSize[1] * program.workPerThread;
  const tileBOuter = program.workGroupSize[0] * program.vecSize;
  const tileInner = tileBOuter;  // Make sure tileInner is divisible by 4.

  const tileSizeA = [tileAOuter, tileInner];
  const tileSizeB = [tileInner, tileBOuter];
  return [
    tilesFitEvenlyIntoShape(tileSizeA, program.aShape.slice(1)),
    tilesFitEvenlyIntoShape(tileSizeB, bShape.slice(1))
  ];
}

export function setProgramUniformForConv2D(
    program: Conv2DMMProgram|Conv2DMMVec4Program,
    dimensions: Array<{type: string; data: number[]}>) {
  const tileAOuter = program.workGroupSize[1] * program.elementsPerThread[1];
  const tileBOuter = program.workGroupSize[0] * program.elementsPerThread[0];
  let tileInner;
  if (program instanceof Conv2DMMProgram) {
    tileInner = tileAOuter > tileBOuter ? tileAOuter : tileBOuter;
    util.assert(
        tileInner % program.workGroupSize[0] === 0 &&
            tileInner % program.workGroupSize[1] === 0,
        () =>
            // tslint:disable-next-line: max-line-length
        'tileInner must be multiple of workgroupsize.x and workgroupsize.y');
  } else {
    tileInner = tileBOuter;
  }
  const tileSizeA = [tileAOuter, tileInner];
  const tileSizeB = [tileInner, tileBOuter];
  const dimAOuter = program.outputShape[1] * program.outputShape[2];
  const dimBOuter = program.outputShape[3];
  const dimInner = program.convInfo.filterHeight *
      program.convInfo.filterWidth * program.convInfo.inChannels;

  const fitA = tilesFitEvenlyIntoShape(tileSizeA, [dimAOuter, dimInner]);
  const fitB = tilesFitEvenlyIntoShape(tileSizeB, [dimInner, dimBOuter]);
  dimensions.push(
      {type: 'int32', data: [dimAOuter]}, {type: 'int32', data: [dimBOuter]},
      {type: 'int32', data: [dimInner]}, {type: 'int32', data: [Number(fitA)]},
      {type: 'int32', data: [Number(fitB)]});
}
