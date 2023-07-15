/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {getMainHeaderString as main, PixelsOpType, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class FromPixelsProgram implements WebGPUProgram {
  dispatch: [number, number, number];
  dispatchLayout: {x: number[]};
  pixelsOpType = PixelsOpType.FROM_PIXELS;
  outputShape: number[] = [0];
  shaderKey: string;
  importVideo: boolean;
  variableNames: string[] = [];
  workgroupSize: [number, number, number] =
      [256, 1, 1];  // The empirical value.

  constructor(outputShape: number[], numChannels: number, importVideo = false) {
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize,
        [numChannels, 1, 1]);

    this.importVideo = importVideo;
    this.shaderKey = `fromPixels_${this.importVideo}`;
  }

  getUserCode(): string {
    const textureLoad = this.importVideo ?
        'textureLoad(src, vec2<i32>(coords.yx));' :
        'textureLoad(src, vec2<i32>(coords.yx), 0)';
    const textureType =
        this.importVideo ? 'texture_external' : 'texture_2d<f32>';
    return `
      @binding(1) @group(0) var src: ${textureType};
      ${main('index')} {
        let flatIndex = index * uniforms.numChannels;
        if (flatIndex < uniforms.size) {
          let coords = getCoordsFromIndex(flatIndex);
          let values = ${textureLoad};
          for (var i = 0; i < uniforms.numChannels; i = i + 1) {
            result[flatIndex + i] = i32(floor(255.0 * values[i]));
          }
        }
      }
  `;
  }
}
