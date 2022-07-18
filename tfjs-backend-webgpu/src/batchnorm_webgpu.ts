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
import {getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class BatchNormProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y?: number[], z?: number[]};
  dispatch: [number, number, number];
  variableNames: string[];
  uniforms = 'varianceEpsilon : f32,';
  // This is an experimental value.
  workGroupSize: [number, number, number] = [128, 1, 1];
  offsetShape: number[]|null;
  scaleShape: number[]|null;
  varianceEpsilon: number;
  size = true;

  constructor(
      xShape: number[], meanShape: number[], varianceShape: number[],
      offsetShape: number[]|null, scaleShape: number[]|null) {
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
    this.shaderKey = 'batchNorm';
  }

  getUserCode(): string {
    let offsetSnippet = '0.0';
    if (this.offsetShape != null) {
      offsetSnippet = 'getOffsetByOutputIndex(index)';
    }

    let scaleSnippet = '1.0';
    if (this.scaleShape != null) {
      scaleSnippet = 'getScaleByOutputIndex(index)';
    }

    const userCode = `
      ${getMainHeaderAndGlobalIndexString()}
        if (index < uniforms.size)
        {
          let xValue = getXByOutputIndex(index);
          let meanValue = getMeanByOutputIndex(index);
          let varianValue = getVarianceByOutputIndex(index);
          let offsetValue = ${offsetSnippet};
          let scaleValue = ${scaleSnippet};
          let inv = scaleValue * inverseSqrt(varianValue + f32(uniforms.varianceEpsilon));
          setOutputAtIndex(index,dot(vec3<f32>(xValue, -meanValue, offsetValue), vec3<f32>(inv, inv, 1.0)));
        }
      }
  `;
    return userCode;
  }
}
