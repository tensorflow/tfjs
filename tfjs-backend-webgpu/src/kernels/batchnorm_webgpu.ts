/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {getCoordsDataType} from '../shader_preprocessor';
import {computeDispatch} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class BatchNormProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames: string[];
  workGroupSize: [4, 4, 4];
  needsShapesUniforms = true;

  constructor(
      xShape: number[], meanShape: number[], varianceShape: number[],
      offsetShape: number[]|null, scaleShape: number[]|null,
      varianceEpsilon: number) {
    this.variableNames = ['x', 'mean', 'variance'];
    backend_util.assertAndGetBroadcastShape(xShape, meanShape);
    backend_util.assertAndGetBroadcastShape(xShape, varianceShape);
    this.outputShape = xShape;
    this.dispatchLayout = {x: [1, 2], y: [0], z: [3]};
    const dim = this.outputShape.length;
    const coordsDataType = getCoordsDataType(dim);
    let setOutput =
        'setOutput(coords[0], coords[1], coords[2], coords[3], value);';
    if (dim === 2) {
      this.dispatchLayout = {x: [1], y: [0], z: []};
      setOutput = 'setOutput(coords[0], coords[1], value);';
    }
    if (dim === 3) {
      this.dispatchLayout = {x: [1, 2], y: [0], z: []};
      setOutput = 'setOutput(coords[0], coords[1], coords[2], value);';
    }
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    let offsetSnippet = '0.0';
    if (offsetShape != null) {
      backend_util.assertAndGetBroadcastShape(xShape, offsetShape);
      this.variableNames.push('offset');
      offsetSnippet = 'getOffsetAtOutCoords()';
    }

    let scaleSnippet = '1.0';
    if (scaleShape != null) {
      backend_util.assertAndGetBroadcastShape(xShape, scaleShape);
      this.variableNames.push('scale');
      scaleSnippet = 'getScaleAtOutCoords()';
    }

    this.userCode = `
      void writeResult(${coordsDataType} coords,float value) {
        if (coordsInBounds(coords, outShape)) {
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
        float inv = scale * inversesqrt(variance + float(${varianceEpsilon}));
        writeResult(coords,dot(vec3(x, -mean, offset), vec3(inv, inv, 1)));
      }
  `;
  }
}
