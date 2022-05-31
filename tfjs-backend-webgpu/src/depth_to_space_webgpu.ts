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

import {getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class DepthToSpaceProgram implements WebGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  dataFormat: string;
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [64, 1, 1];
  size = true;
  uniforms = 'blockSize : i32,';

  constructor(outputShape: number[], dataFormat: 'NHWC'|'NCHW') {
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.shaderKey = `depthToSpace_${dataFormat}`;
    this.dataFormat = dataFormat;
  }

  getUserCode(): string {
    const userCode = `
      ${getMainHeaderAndGlobalIndexString()}
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let b = coords[0];
          let h = ${this.getHeightCoordString()};
          let w = ${this.getWidthCoordString()};
          let d = ${this.getDepthCoordString()};

          let in_h = h / uniforms.blockSize;
          let offset_h = h % uniforms.blockSize;
          let in_w = w / uniforms.blockSize;
          let offset_w = w % uniforms.blockSize;
          let offset_d = (offset_h * uniforms.blockSize + offset_w) *
            ${this.getOutputDepthSize()};
          let in_d = d + offset_d;

          let rlt = ${this.getInputSamplingString()};
          setOutputAtIndex(index, rlt);
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

  private getOutputDepthSize(): string {
    if (this.dataFormat === 'NHWC') {
      return `uniforms.outShape[3]`;
    } else {
      return `uniforms.outShape[1]`;
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
