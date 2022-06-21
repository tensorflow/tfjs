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
import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';

export class GatherNDProgram implements GPGPUProgram {
  variableNames = ['x', 'indices'];
  outputShape: number[];
  userCode: string;
  constructor(
      private sliceDim: number, private strides: number[], shape: number[],
      private paramsShape: number[]) {
    this.outputShape = shape;
    const stridesType = getCoordsDataType(strides.length);
    const dtype = getCoordsDataType(shape.length);
    const strideString = this.sliceDim > 1 ? 'strides[j]' : 'strides';
    const paramsShapeType = getCoordsDataType(paramsShape.length);
    const paramsShapeString = paramsShape.length > 1 ? 'paramsShape[j]' : 'paramsShape';
    this.userCode = `
        ${stridesType} strides = ${stridesType}(${this.strides});
        ${paramsShapeType} paramsShape = ${paramsShapeType}(${this.paramsShape});
         void main() {
          ${dtype} coords = getOutputCoords();
          int flattenIndex = 0;
          bool out_of_bounds = false;
          for (int j = 0; j < ${this.sliceDim}; j++) {
            int index = round(getIndices(coords[0], j));
            out_of_bounds = out_of_bounds || index < 0;
            out_of_bounds = out_of_bounds || index >= ${paramsShapeString};
            flattenIndex += index * ${strideString};
          }
          setOutput(out_of_bounds ? 0.0 : getX(flattenIndex, coords[1]));
        }
      `;
  }
}
