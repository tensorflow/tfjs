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

import * as concat_util from '../../ops/concat_util';
import {getChannels} from '../packing_util';

import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';

export class ConcatPackedProgram implements GPGPUProgram {
  variableNames: string[];
  usesPackedTextures = true;
  outputShape: number[] = [];
  userCode: string;

  constructor(shapes: number[][], axis: number) {
    this.outputShape = concat_util.computeOutShape(shapes, axis);
    const shape = this.outputShape;
    const rank = shape.length;
    const dtype = getCoordsDataType(rank);
    const coords = getChannels('coords', rank);
    const channels = ['x', 'y', 'z', 'w', 'u', 'v'].slice(0, rank);
    this.variableNames = shapes.map((_, i) => `T${i}`);

    const offsets: number[] = new Array(shapes.length - 1);
    offsets[0] = shapes[0][axis];
    for (let i = 1; i < offsets.length; i++) {
      offsets[i] = offsets[i - 1] + shapes[i][axis];
    }

    const channel = channels[axis];
    const lastChannels = 'vec2(' + channels.slice(-2).join() + ')';
    const allChannels = channels.join();

    let getValueSnippet = `if (${channel} < ${offsets[0]})
          return getChannel(getT0(${allChannels}), ${lastChannels});`;
    for (let i = 1; i < offsets.length; i++) {
      const shift = offsets[i - 1];
      getValueSnippet += `
        else if (${channel} < ${offsets[i]}) {
          ${channel} -= ${shift};
          return getChannel(getT${i}(${allChannels}), ${lastChannels});
        }`;
    }
    const lastIndex = offsets.length;
    const shift = offsets[offsets.length - 1];
    getValueSnippet += `
        else {
          ${channel} -= ${shift};
          return getChannel(getT${lastIndex}(${allChannels}), ${lastChannels});
        }`;

    this.userCode = `
      float getValue(${channels.map(x => 'int ' + x)}) {
        ${getValueSnippet}
      }

      void main() {
        ${dtype} coords = getOutputCoords();
        vec4 result = vec4(getValue(${coords}), 0., 0., 0.);
        if (++${coords[rank - 1]} < ${shape[rank - 1]}) {
          result.g = getValue(${coords});
        }
        if (++${coords[rank - 2]} < ${shape[rank - 2]}) {
          result.a = getValue(${coords});
        }
        if (${coords[rank - 2]} < ${shape[rank - 2]} &&
            --${coords[rank - 1]} < ${shape[rank - 1]}) {
          result.b = getValue(${coords});
        }
        setOutput(result);
      }
    `;
  }
}
