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

import {getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class FromPixelsProgram implements WebGPUProgram {
  outputShape: number[] = [0];
  shaderKey: string;
  workPerThread: number;
  dispatchLayout: {x: number[]};
  variableNames: string[] = [];
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] =
      [256, 1, 1];  // The empirical value.

  useImport: boolean;

  constructor(outputShape: number[], useImport = false) {
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.useImport = useImport;
    this.shaderKey = `fromPixels_${this.useImport}`;
  }

  getUserCode(): string {
    const textureLoad = this.useImport ?
        'textureLoad(src, vec2<i32>(coords.yx));' :
        'textureLoad(src, vec2<i32>(coords.yx), 0)';
    const textureType = this.useImport ? 'texture_external' : 'texture_2d<f32>';
    return `
      @binding(1) @group(0) var src: ${textureType};

      ${getMainHeaderAndGlobalIndexString()}
        let flatIndexBase = index * uniforms.numChannels;
        for (var i = 0; i < uniforms.numChannels; i = i + 1) {
          let flatIndex = flatIndexBase + i;
          if (flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndexBase);
            let values = ${textureLoad};
            result[flatIndex] = i32(floor(255.0 * values[i]));
          }
        }
      }
  `;
  }
}
