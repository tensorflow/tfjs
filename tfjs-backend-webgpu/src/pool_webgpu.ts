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
import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class Pool2DProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  uniforms =
      `strides : vec2<i32>, pads : vec2<i32>, dilations : vec2<i32>, convDims : vec2<i32>, filterDims : vec2<i32>,`;
  // TODO(jiajia.qin@intel.com): Dynamically choose different workgroupSize for
  // different output shapes.
  workgroupSize: [number, number, number] = [128, 1, 1];
  poolType: 'max'|'avg';
  size = true;
  computePositions: boolean;
  flattenPositions: boolean;
  includeBatchIndex: boolean;

  constructor(
      convInfo: backend_util.Conv2DInfo, poolType: 'max'|'avg',
      computePositions = false, flattenPositions = false,
      includeBatchIndex = false) {
    if (poolType === 'avg' && computePositions) {
      throw new Error('Cannot compute positions for average pool.');
    }

    this.outputShape = convInfo.outShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);

    this.poolType = poolType;
    this.computePositions = computePositions;
    this.flattenPositions = flattenPositions;
    this.includeBatchIndex = includeBatchIndex;
    this.shaderKey = `pool2D_${poolType}_${computePositions}_${
        flattenPositions}_${includeBatchIndex}`;
  }

  getUserCode(): string {
    let updateSnippet: string;
    if (this.poolType === 'avg') {
      updateSnippet = `resultValue = resultValue + value; count = count + 1.0;`;
    } else if (this.computePositions) {
      const positionStr = this.flattenPositions ?
          (this.includeBatchIndex ?
               `((batch * uniforms.xShape[1] + xR) * uniforms.xShape[2] + xC) * uniforms.xShape[3] + d` :
               `(xR * uniforms.xShape[2] + xC) * uniforms.xShape[3] + d`) :
          `wR * uniforms.filterDims.y + wC`;
      updateSnippet = `let currMaxValue = mix(value, maxValue, maxValueFound);
      if (value >= currMaxValue) {
        maxValue = value;
        maxValueFound = 1.0;
        maxPosition = ${positionStr};
      }`;
    } else {
      updateSnippet = `resultValue = max(value, resultValue);`;
    }

    let returnValue = `resultValue`;
    if (this.poolType === 'avg') {
      returnValue = `resultValue / max(count, 1.0)`;
    }

    const userCode = `
      ${main('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
          let batch = coords[0];
          let d = coords[3];
          let xRCCorner = vec2<i32>(coords.yz) * uniforms.strides - uniforms.pads;
          let xRCorner = xRCCorner.x;
          let xCCorner = xRCCorner.y;

          ${
        this.computePositions ?
            `var maxValue = 0.0;
            var maxValueFound = 0.0;
            var maxPosition = 0;` :
            `var resultValue = ${
                this.poolType === 'avg' ? '0.0' : '-1.0 / pow(10.0, -20.0)'};`}

          var count = 0.0;
          for (var wR = 0; wR < uniforms.filterDims.x; wR = wR + uniforms.dilations.x) {
            let xR = xRCorner + wR;

            if (xR < 0 || xR >= uniforms.convDims.x) {
              continue;
            }

            for (var wC = 0; wC < uniforms.filterDims.y; wC = wC + uniforms.dilations.y) {
              let xC = xCCorner + wC;
              if (xC < 0 || xC >= uniforms.convDims.y) {
                continue;
              }

              let value = getX(batch, xR, xC, d);
              ${updateSnippet}
            }
          }

          ${
        this.computePositions ? `setOutputAtIndexI32(index, maxPosition);` :
                                `setOutputAtIndex(index, ${returnValue});`}
        }
      }
    `;
    return userCode;
  }
}

export class Pool3DProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  uniforms =
      `strides : vec3<i32>, pads : vec3<i32>, convDims : vec3<i32>, filterDims : vec3<i32>,`;
  workgroupSize: [number, number, number] = [128, 1, 1];
  poolType: 'max'|'avg';
  size = true;
  computePositions: boolean;
  flattenPositions: boolean;
  includeBatchIndex: boolean;

  constructor(
      convInfo: backend_util.Conv3DInfo, poolType: 'max'|'avg',
      computePositions = false, flattenPositions = false,
      includeBatchIndex = false) {
    if (poolType === 'avg' && computePositions) {
      throw new Error('Cannot compute positions for average pool.');
    }

    this.outputShape = convInfo.outShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);

    this.poolType = poolType;
    this.computePositions = computePositions;
    this.flattenPositions = flattenPositions;
    this.includeBatchIndex = includeBatchIndex;
    this.shaderKey = `pool3D_${poolType}_${computePositions}_${
        flattenPositions}_${includeBatchIndex}`;
  }

  getUserCode(): string {
    let updateSnippet: string;
    if (this.poolType === 'avg') {
      updateSnippet = `resultValue += value; count += 1.0;`;
    } else if (this.computePositions) {
      const positionStr = this.flattenPositions ?
          (this.includeBatchIndex ?
               `(((batch * uniforms.xShape.y + xD) * uniforms.xShape.z + xR) * uniforms.xShape.w + xC) * uniforms.xShape.u + ch` :
               `((xD * uniforms.xShape.z + xR) * uniforms.xShape.w + xC) * uniforms.xShape.u + ch`) :
          `wD * uniforms.filterDims.y * uniforms.filterDims.y + wR * uniforms.filterDims.z + wC`;
      updateSnippet = `let currMaxValue = mix(value, maxValue, maxValueFound);
      if (value >= currMaxValue) {
        maxValue = value;
        maxValueFound = 1.0;
        maxPosition = ${positionStr};
      }`;
    } else {
      updateSnippet = `resultValue = max(value, resultValue);`;
    }

    let returnValue = `resultValue`;
    if (this.poolType === 'avg') {
      returnValue = `resultValue / max(count, 1.0)`;
    }

    const userCode = `
      ${main('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let batch = coords.x;
          let ch = coords.u;

          let xCorner = vec3<i32>(coords.y, coords.z, coords.w) * uniforms.strides - uniforms.pads;
          let xDCorner = xCorner.x;
          let xRCorner = xCorner.y;
          let xCCorner = xCorner.z;

          ${
        this.computePositions ?
            `var maxValue = 0.0;
            var maxValueFound = 0.0;
            var maxPosition = 0;` :
            `var resultValue = ${
                this.poolType === 'avg' ? '0.0' : '-1.0 / pow(10.0, -20.0)'};`}

          var count = 0.0;
          for (var wD = 0; wD < uniforms.filterDims.x; wD++) {
            let xD = xDCorner + wD;
            if (xD < 0 || xD >= uniforms.convDims.x) {
              continue;
            }

            for (var wR = 0; wR < uniforms.filterDims.y; wR++) {
              let xR = xRCorner + wR;
              if (xR < 0 || xR >= uniforms.convDims.y) {
                continue;
              }

              for (var wC = 0; wC < uniforms.filterDims.z; wC++) {
                let xC = xCCorner + wC;
                if (xC < 0 || xC >= uniforms.convDims.z) {
                  continue;
                }

                let value = getX(batch, xD, xR, xC, ch);
                ${updateSnippet}
              }
            }
          }

          ${
        this.computePositions ? `setOutputAtIndexI32(index, maxPosition);` :
                                `setOutputAtIndex(index, ${returnValue});`}
        }
      }
    `;
    return userCode;
  }
}
