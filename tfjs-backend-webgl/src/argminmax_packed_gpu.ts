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

import {util} from '@tensorflow/tfjs-core';

import {GPGPUProgram} from './gpgpu_math';
import {getChannels} from './packing_util';
import {getCoordsDataType} from './shader_compiler';

export class ArgMinMaxPackedProgram implements GPGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  userCode: string;
  packedInputs = true;
  packedOutput = true;

  constructor(
      shape: number[], windowSize: number, op: 'max'|'min',
      firstPass: boolean) {
    util.assert(
        shape.length > 2,
        () => `Packed arg${
            op.charAt(0).toUpperCase() +
            op.slice(1)} supports only inputs with rank above 2.`);
    const inSize = shape[shape.length - 1];
    const outSize = Math.ceil(inSize / windowSize);
    this.outputShape = shape.slice(0, -1);
    if (outSize > 1) {
      this.outputShape.push(outSize);
    }
    if (!firstPass) {
      this.variableNames.push('bestIndicesA');
    }
    const outShape = this.outputShape;
    const rank = outShape.length;
    const dtype = getCoordsDataType(rank);
    const coords = getChannels('coords', rank);

    let sourceLocSetup;
    let sourceRank;
    if (outSize === 1) {
      sourceRank = rank + 1;
      const sourceLocDType = getCoordsDataType(sourceRank);
      sourceLocSetup = `
        ${sourceLocDType} sourceLocR = ${sourceLocDType}(${coords.join()}, 0);
        ++${coords[rank - 1]};
        ${sourceLocDType} sourceLocG = ${sourceLocDType}(${coords.join()}, 0);
        ++${coords[rank - 2]};
        ${sourceLocDType} sourceLocA = ${sourceLocDType}(${coords.join()}, 0);
        --${coords[rank - 1]};
        ${sourceLocDType} sourceLocB = ${sourceLocDType}(${coords.join()}, 0);
        --${coords[rank - 2]};`;
    } else {
      sourceRank = rank;
      sourceLocSetup = `
        ${dtype} sourceLocR = coords;
        ++${coords[rank - 1]};
        ${dtype} sourceLocG = coords;
        ++${coords[rank - 2]};
        ${dtype} sourceLocA = coords;
        --${coords[rank - 1]};
        ${dtype} sourceLocB = coords;
        --${coords[rank - 2]};`;
    }
    const channels = ['x', 'y', 'z', 'w', 'u', 'v'].slice(0, sourceRank);
    const inChannel = '.' + channels[sourceRank - 1];  // e.g. ".b" for rank 3.
    const intChannels = channels.map(x => 'int ' + x);
    const srcRCoords =
        getChannels('sourceLocR', sourceRank - 1).concat('inIdx.r');
    const srcGCoords =
        getChannels('sourceLocG', sourceRank - 1).concat('inIdx.g');
    const srcBCoords =
        getChannels('sourceLocB', sourceRank - 1).concat('inIdx.b');
    const srcACoords =
        getChannels('sourceLocA', sourceRank - 1).concat('inIdx.a');

    const compOp = (op === 'max') ? 'greaterThan' : 'lessThan';
    const fetchCandidateIdx = firstPass ? '' : `
          inIdx = round(vec4(getBestIndicesAChannel(${srcRCoords.join()}),
                             getBestIndicesAChannel(${srcGCoords.join()}),
                             getBestIndicesAChannel(${srcBCoords.join()}),
                             getBestIndicesAChannel(${srcACoords.join()})));`;

    const fetchValue = `vec4(
            getAChannel(${srcRCoords.join()}),
            hasNextCol ? getAChannel(${srcGCoords.join()}) : 0.,
            hasNextRow ? getAChannel(${srcBCoords.join()}) : 0.,
            hasNextRow && hasNextCol ? getAChannel(${srcACoords.join()}) : 0.)`;

    const getBestIndicesAChannelSnippet = firstPass ? '' : `
      float getBestIndicesAChannel(${intChannels.join()}) {
        return getChannel(getBestIndicesA(${channels.join()}),
                                          vec2(${channels.slice(-2).join()}));
      }`;

    this.userCode = `
      float getAChannel(${intChannels.join()}) {
        return getChannel(getA(${channels.join()}),
                               vec2(${channels.slice(-2).join()}));
      }
      ${getBestIndicesAChannelSnippet}
      void main() {
        ${dtype} coords = getOutputCoords();
        bool hasNextCol = ${coords[rank - 1]} < ${outShape[rank - 1] - 1};
        bool hasNextRow = ${coords[rank - 2]} < ${outShape[rank - 2] - 1};
        ${sourceLocSetup}
        ivec4 srcIdx = ivec4(sourceLocR${inChannel}, sourceLocG${inChannel},
          sourceLocB${inChannel}, sourceLocA${inChannel}) * ${windowSize};
        ivec4 inIdx = srcIdx;
        vec4 bestIndex = vec4(inIdx);
        vec4 bestValue = ${fetchValue};

        for (int i = 0; i < ${windowSize}; i++) {
          inIdx = srcIdx;
          ${fetchCandidateIdx}
          vec4 candidate = ${fetchValue};
          bvec4 nan = isnan(candidate);
          bvec4 replace = bvec4(
            vec4(${compOp}(candidate, bestValue)) * (vec4(1.0) - vec4(nan)));

          bestValue = vec4(replace.x  ? candidate.x : bestValue.x,
                           replace.y  ? candidate.y : bestValue.y,
                           replace.z  ? candidate.z : bestValue.z,
                           replace.w  ? candidate.w : bestValue.w);
          bestIndex = mix(bestIndex, vec4(inIdx), vec4(replace));
          srcIdx++;
        }
        setOutput(bestIndex);
      }
    `;
  }
}
