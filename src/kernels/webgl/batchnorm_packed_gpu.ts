/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as broadcast_util from '../../ops/broadcast_util';
import {GPGPUProgram} from './gpgpu_math';

export class BatchNormPackedProgram implements GPGPUProgram {
  variableNames: string[];
  outputShape: number[];
  userCode: string;
  supportsBroadcasting = true;
  usesPackedTextures = true;

  constructor(
      xShape: number[], meanShape: number[], varianceShape: number[],
      offsetShape: number[]|null, scaleShape: number[]|null,
      varianceEpsilon: number) {
    this.variableNames = ['x', 'mean', 'variance'];
    broadcast_util.assertAndGetBroadcastShape(xShape, meanShape);
    broadcast_util.assertAndGetBroadcastShape(xShape, varianceShape);

    const meanSnippet = broadcastSample('mean', meanShape.length);
    const varianceSnippet = broadcastSample('variance', varianceShape.length);

    let offsetSnippet = 'vec4 offset = vec4(0.0)';
    if (offsetShape != null) {
      broadcast_util.assertAndGetBroadcastShape(xShape, offsetShape);
      this.variableNames.push('offset');
      offsetSnippet = broadcastSample('offset', offsetShape.length);
    }

    let scaleSnippet = 'vec4 scale = vec4(1.0)';
    if (scaleShape != null) {
      broadcast_util.assertAndGetBroadcastShape(xShape, scaleShape);
      this.variableNames.push('scale');
      scaleSnippet = broadcastSample('scale', scaleShape.length);
    }

    this.outputShape = xShape;
    this.userCode = `
      void main() {
        ivec4 rc = getOutputCoords();

        ${offsetSnippet};
        ${scaleSnippet};

        vec4 x = getX(rc.x, rc.y, rc.z, rc.w);
        ${meanSnippet};
        ${varianceSnippet};

        vec4 inv = scale * inversesqrt(variance + vec4(${varianceEpsilon}));

        setOutput((x - mean) * inv + offset);
      }
    `;
  }
}

function broadcastSample(texName: string, rank: number): string {
  const texSampler = `get${texName.charAt(0).toUpperCase()}${texName.slice(1)}`;
  if (rank === 1) {
    return `
      vec4 ${texName}Sample = ${texSampler}(rc.w);
      vec4 ${texName} = vec4(${texName}Sample.xy, ${texName}Sample.xy);
    `;
  }
  return `vec4 ${texName} = ${texSampler}(rc.x, rc.y, rc.z, rc.w)`;
}