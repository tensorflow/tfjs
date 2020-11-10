/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
import {GPGPUProgram} from './gpgpu_math';

export class BatchNormProgram implements GPGPUProgram {
  variableNames: string[];
  outputShape: number[] = [];
  userCode: string;

  constructor(
      xShape: number[], meanShape: number[], varianceShape: number[],
      offsetShape: number[]|null, scaleShape: number[]|null,
      varianceEpsilon: number) {
    this.variableNames = ['x', 'mean', 'variance'];
    backend_util.assertAndGetBroadcastShape(xShape, meanShape);
    backend_util.assertAndGetBroadcastShape(xShape, varianceShape);

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

    this.outputShape = xShape;
    this.userCode = `
      void main() {
        float x = getXAtOutCoords();
        float mean = getMeanAtOutCoords();
        float variance = getVarianceAtOutCoords();
        float offset = ${offsetSnippet};
        float scale = ${scaleSnippet};
        float inv = scale * inversesqrt(variance + float(${varianceEpsilon}));
        setOutput(dot(vec3(x, -mean, offset), vec3(inv, inv, 1)));
      }
    `;
  }
}
