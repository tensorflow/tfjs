/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {GPGPUProgram} from './gpgpu_math';
import {getChannels} from './packing_util';
import {getCoordsDataType} from './shader_compiler';

/**
 * Example shader code for
 * `mirrorPad(tf.tensor1d([1, 2, 3], 'int32'), [[2, 2]], 'reflect')`
 * ```
 *    const int start = int(2);
 *    const int end = int(5);
 *
 *    void main() {
 *       int outputLoc = getOutputCoords();
 *       vec4 result = vec4(0.);
 *
 *       int rc = outputLoc;
 *
 *       int source = rc;
 *       if (source < start) {
 *         source = start * 2 - source - 0;
 *       } else if (source >= end) {
 *         source = (end - 1) * 2 - source + 0;
 *       }
 *       source -= start;
 *
 *       result[0] = getChannel(getX(source), source);
 *       rc += 1;
 *       if(rc < 6) {
 *          int source = rc;
 *          if (source < start) {
 *            source = start * 2 - source - 0;
 *          } else if (source >= end) {
 *            source = (end - 1) * 2 - source + 0;
 *          }
 *          source -= start;
 *
 *         result[1] = getChannel(getX(source), source);
 *       }
 *
 *       setOutput(result);
 *     }
 * ```
 */
export class MirrorPadPackedProgram implements GPGPUProgram {
  variableNames = ['x'];
  packedInputs = true;
  packedOutput = true;
  outputShape: number[];
  userCode: string;

  constructor(
      xShape: number[], paddings: Array<[number, number]>,
      mode: 'reflect'|'symmetric') {
    this.outputShape = paddings.map(
        (p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
    const rank = xShape.length;
    const dtype = getCoordsDataType(rank);

    const start = paddings.map(p => p[0]).join(',');
    const end = paddings.map((p, i) => p[0] + xShape[i]).join(',');
    const coords = getChannels('rc', rank);
    const source = getChannels('source', rank);
    const cLimit = `${coords[rank - 1]} < ${this.outputShape[rank - 1]}`;
    const innerDims =
        rank === 1 ? 'source' : `vec2(${source.slice(-2).join()})`;
    const offset = mode === 'reflect' ? 0 : 1;

    let mainLoop = '';
    if (rank === 1) {
      const padSetup = `
        ${dtype} source = rc;
        if (source < start) {
          source = start * 2 - source - ${offset};
        } else if (source >= end) {
          source = (end - 1) * 2 - source + ${offset};
        }
        source -= start;
      `;
      mainLoop = `
        ${dtype} rc = outputLoc;
        ${padSetup}
        result[0] = getChannel(getX(${source.join()}), ${innerDims});
        ${coords[rank - 1]} += 1;
        if(${cLimit}) {
          ${padSetup}
          result[1] = getChannel(getX(${source.join()}), ${innerDims});
        }
      `;
    } else {
      const padSetup = `
        ${dtype} source = rc;
        ${dtype} lt = ${dtype}(lessThan(source, start));
        ${dtype} gte = ${dtype}(greaterThanEqual(source, end));
        ${dtype} orig = 1 - (lt + gte);
        source = orig * source +
                lt * (start * 2 - source - ${offset}) +
                gte * ((end - 1) * 2 - source + ${offset});
        source -= start;
      `;

      mainLoop = `
        ${dtype} rc = outputLoc;
        ${padSetup}
        result[0] = getChannel(getX(${source.join()}), ${innerDims});
        ${coords[rank - 1]} += 1;
        if(${cLimit}) {
          ${padSetup}
          result[1] = getChannel(getX(${source.join()}), ${innerDims});
        }
        rc = outputLoc;
        ${coords[rank - 2]} += 1;
        if(${coords[rank - 2]} < ${this.outputShape[rank - 2]}) {
          ${padSetup}
          result[2] = getChannel(getX(${source.join()}), ${innerDims});
          ${coords[rank - 1]} += 1;
          if(${cLimit}) {
            ${padSetup}
            result[3] = getChannel(getX(${source.join()}), ${innerDims});
          }
        }
      `;
    }

    this.userCode = `
      const ${dtype} start = ${dtype}(${start});
      const ${dtype} end = ${dtype}(${end});

      void main() {
        ${dtype} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${mainLoop}
        setOutput(result);
      }
    `;
  }
}
