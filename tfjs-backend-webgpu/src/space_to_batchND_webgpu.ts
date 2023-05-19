/**
 * @license
 * Copyright 2023 Google LLC.
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

import {padCommon} from './pad_webgpu';
import {getSwitchedCoords} from './transpose_webgpu';
import {getCoordsDataType, getCoordsFromIndexSnippet, getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class SpaceToBatchNDProgram implements WebGPUProgram {
  variableNames = ['x'];
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  uniforms = '';
  workgroupSize: [number, number, number] = [64, 1, 1];
  newDim: number[];
  xShape: number[];
  paddedXShape: number[];
  size = true;

  constructor(
      xShape: number[], paddedXShape: number[],
      paddings: Array<[number, number]>, reshapedPaddedXShape: number[],
      newDim: number[], paddedXShapeStridesShapeLength: number) {
    const outputShape: number[] = new Array(reshapedPaddedXShape.length);
    for (let i = 0; i < outputShape.length; i++) {
      outputShape[i] = reshapedPaddedXShape[newDim[i]];
    }
    this.outputShape = outputShape;
    this.newDim = newDim;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);
    this.xShape = xShape;
    this.paddedXShape = paddedXShape;
    this.uniforms += `reshapedPaddedXShape : ${
        getCoordsDataType(
            reshapedPaddedXShape.length)}, paddedXShapeStrides : ${
        getCoordsDataType(paddedXShapeStridesShapeLength)}, `;
    paddings.map((_, i) => {
      this.uniforms += ` pad${i} : vec2<i32>,`;
    });
    this.shaderKey = `spaceToBatchND_${newDim}`;
  }

  getUserCode(): string {
    const dtype = getCoordsDataType(this.outputShape.length);
    const switched = getSwitchedCoords(this.newDim);

    const userCode = `
      ${getCoordsFromIndexSnippet(this.paddedXShape, 'PaddedX')}
      ${main('index')} {
        if(index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let switchedIndex = getIndexFromCoords${this.outputShape.length}D(${
        dtype}(${switched}), uniforms.reshapedPaddedXShape);
          let paddedCoords = getPaddedXCoordsFromIndex(switchedIndex);
          ${padCommon(this.xShape, true)}
        }
      }
    `;
    return userCode;
  }
}
