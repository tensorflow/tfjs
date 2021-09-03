/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {getGlobalIndexStringWgsl, getMainHeaderStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class DepthToSpaceProgram implements WebGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  blockSize: number;
  dataFormat: string;
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [64, 1, 1];
  size: number;
  useWgsl: boolean;

  constructor(
      outputShape: number[], blockSize: number, dataFormat: 'NHWC'|'NCHW') {
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.shaderKey = `depthToSpace_${blockSize}_${dataFormat}`;
    this.size = util.sizeFromShape(this.outputShape);
    this.blockSize = blockSize;
    this.dataFormat = dataFormat;
  }

  getUserCode(): string {
    const userCode = `
      void main() {
        int index = getGlobalIndex();
        if (index < size) {
          ivec4 coords = getOutputCoords();
          int b = coords[0];
          int h = ${this.getHeightCoordString()};
          int w = ${this.getWidthCoordString()};
          int d = ${this.getDepthCoordString()};

          int in_h = h / ${this.blockSize};
          int offset_h = h % ${this.blockSize};
          int in_w = w / ${this.blockSize};
          int offset_w = w % ${this.blockSize};
          int offset_d = (offset_h * ${this.blockSize} + offset_w) *
            ${this.getOutputDepthSize()};
          int in_d = d + offset_d;

          float result = ${this.getInputSamplingString()};
          setOutput(index, result);
        }
      }`;
    return userCode;
  }

  getUserCodeWgsl(): string {
    const userCode = `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let index = getGlobalIndex();
        if (index < size) {
          let coords = getOutputCoords();
          let b = coords[0];
          let h = ${this.getHeightCoordString()};
          let w = ${this.getWidthCoordString()};
          let d = ${this.getDepthCoordString()};

          let in_h = h / ${this.blockSize};
          let offset_h = h % ${this.blockSize};
          let in_w = w / ${this.blockSize};
          let offset_w = w % ${this.blockSize};
          let offset_d = (offset_h * ${this.blockSize} + offset_w) *
            ${this.getOutputDepthSize()};
          let in_d = d + offset_d;

          let result = ${this.getInputSamplingString()};
          setOutput(index, result);
        }
      }`;
    return userCode;
  }


  private getHeightCoordString(): string {
    if (this.dataFormat === 'NHWC') {
      return `coords[1]`;
    } else {
      return `coords[2]`;
    }
  }

  private getWidthCoordString(): string {
    if (this.dataFormat === 'NHWC') {
      return `coords[2]`;
    } else {
      return `coords[3]`;
    }
  }

  private getDepthCoordString(): string {
    if (this.dataFormat === 'NHWC') {
      return `coords[3]`;
    } else {
      return `coords[1]`;
    }
  }

  private getOutputDepthSize(): number {
    if (this.dataFormat === 'NHWC') {
      return this.outputShape[3];
    } else {
      return this.outputShape[1];
    }
  }

  private getInputSamplingString(): string {
    if (this.dataFormat === 'NHWC') {
      return `getX(b, in_h, in_w, in_d)`;
    } else {
      return `getX(b, in_d, in_h, in_w)`;
    }
  }
}
