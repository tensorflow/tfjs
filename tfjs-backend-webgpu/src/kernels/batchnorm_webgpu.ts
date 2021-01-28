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

import {backend_util} from '@tensorflow/tfjs-core';

import {getCoordsDataType, getShapeCoords} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class BatchNormProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y?: number[], z?: number[]};
  dispatch: [number, number, number];
  variableNames: string[];
  // This is an experimental value.
  workGroupSize: [number, number, number] = [128, 1, 1];
  offsetShape: number[]|null;
  scaleShape: number[]|null;
  varianceEpsilon: number;

  constructor(
      xShape: number[], meanShape: number[], varianceShape: number[],
      offsetShape: number[]|null, scaleShape: number[]|null,
      varianceEpsilon: number) {
    this.variableNames = ['x', 'mean', 'variance'];
    backend_util.assertAndGetBroadcastShape(xShape, meanShape);
    backend_util.assertAndGetBroadcastShape(xShape, varianceShape);
    this.outputShape = xShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    if (offsetShape != null) {
      backend_util.assertAndGetBroadcastShape(xShape, offsetShape);
      this.variableNames.push('offset');
    }
    if (scaleShape != null) {
      backend_util.assertAndGetBroadcastShape(xShape, scaleShape);
      this.variableNames.push('scale');
    }
    this.offsetShape = offsetShape;
    this.scaleShape = scaleShape;
    this.varianceEpsilon = varianceEpsilon;
    this.shaderKey = `batchNorm_${varianceEpsilon}`;
  }

  getUserCode(): string {
    let offsetSnippet = '0.0';
    if (this.offsetShape != null) {
      offsetSnippet = 'getOffsetAtOutCoords()';
    }

    let scaleSnippet = '1.0';
    if (this.scaleShape != null) {
      scaleSnippet = 'getScaleAtOutCoords()';
    }

    const dim = this.outputShape.length;
    const coordsDataType = getCoordsDataType(dim);
    let setOutput =
        'setOutput(coords[0], coords[1], coords[2], coords[3], value);';
    if (dim === 2) {
      setOutput = 'setOutput(coords[0], coords[1], value);';
    }
    if (dim === 3) {
      setOutput = 'setOutput(coords[0], coords[1], coords[2], value);';
    }
    const userCode = `
      void writeResult(${coordsDataType} coords,float value) {
        if (coordsInBounds(coords, ${getShapeCoords(this.outputShape)})) {
          ${setOutput}
        }
      }
      void main() {
        ${coordsDataType} coords = getOutputCoords();
        float x = getXAtOutCoords();
        float mean = getMeanAtOutCoords();
        float variance = getVarianceAtOutCoords();
        float offset = ${offsetSnippet};
        float scale = ${scaleSnippet};
        float inv = scale * inversesqrt(variance + float(${
        this.varianceEpsilon}));
        writeResult(coords,dot(vec3(x, -mean, offset), vec3(inv, inv, 1)));
      }
  `;
    return userCode;
  }
}
