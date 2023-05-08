/**
 * @license
 * Copyright 2022 Google LLC.
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
import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

const powOperatorSnippet = `
  var powValue = 0.0;
  let basis = uniforms.bias + uniforms.alpha * sum;
  if (uniforms.beta == 0.5) {
    powValue = inverseSqrt(basis);
  } else if (uniforms.beta == 1.0) {
    powValue = 1.0 / basis;
  } else {
    powValue = exp(log(basis) * (-uniforms.beta));
  }
`;

export class LRNProgram implements WebGPUProgram {
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  uniforms = 'radius : i32, bias : f32, alpha : f32, beta : f32,';
  workgroupSize: [number, number, number] = [64, 1, 1];
  size = true;

  constructor(xShape: number[]) {
    this.outputShape = xShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);
    this.shaderKey = 'lrn';
  }

  getUserCode(): string {
    const userCode = `
    ${main('index')} {
      if (index < uniforms.size) {
        let coords = getOutputCoords();
        let b = coords[0];
        let r = coords[1];
        let c = coords[2];
        let d = coords[3];

        let x = getX(b, r, c, d);
        var sum = 0.0;
        for (var i = -uniforms.radius; i <= uniforms.radius; i = i + 1) {
          let idx = d + i;
          if (idx >= 0 && idx < uniforms.xShape[3]) {
            let z = getX(b, r, c, idx);
            sum = sum + z * z;
          }
        }
        ${powOperatorSnippet}

        setOutputAtIndex(index, x * powValue);
      }
    }
  `;
    return userCode;
  }
}

export class LRNSharedProgram implements WebGPUProgram {
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  uniforms = 'radius : i32, bias : f32, alpha : f32, beta : f32,';
  workgroupSize: [number, number, number] = [256, 1, 1];
  maxAllowRadius = 16;
  elementsPerWorkgroup: number;

  constructor(xShape: number[], radius: number) {
    util.assert(
        radius <= this.maxAllowRadius,
        () => `Radius must be less than or equal to ${
            this.maxAllowRadius}, current radius is ${radius}`);

    this.outputShape = xShape;
    // The reason why not using this.workgroupSize[0] + 2 * maxAllowRadius here
    // is to make sure that there is only one time global memory load access for
    // each thread.
    this.elementsPerWorkgroup = this.workgroupSize[0] - 2 * this.maxAllowRadius;
    this.dispatchLayout = {x: [3], y: [2], z: [0, 1]};
    this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, [
      this.elementsPerWorkgroup, this.workgroupSize[1], this.workgroupSize[2]
    ]);
    this.shaderKey = 'lrn_shared';
  }

  getUserCode(): string {
    const userCode = `
    var <workgroup>lrnSub: array<f32, ${this.workgroupSize[0]}>;
    const elementsPerWorkgroup = ${this.elementsPerWorkgroup};
    const maxAllowRadius = ${this.maxAllowRadius};

    ${main()} {
      let localDepth = i32(localId.x);
      let workgroupDepth = i32(workgroupId.x) * elementsPerWorkgroup;
      let xDepth = workgroupDepth + localDepth - maxAllowRadius;
      let b = i32(globalId.z) / uniforms.xShape[1];
      let r = i32(globalId.z) - b * uniforms.xShape[1];
      let c = i32(globalId.y);
      let d = workgroupDepth + localDepth;

      var x = 0.0;
      if (xDepth >= 0 && xDepth < uniforms.xShape[3]) {
        x = getX(b, r, c, xDepth);
      }
      lrnSub[localDepth] = x;
      workgroupBarrier();

      if (localDepth < elementsPerWorkgroup && d < uniforms.outShape[3]) {
        var sum = 0.0;
        let index = localDepth + maxAllowRadius;
        for (var i = -uniforms.radius; i <= uniforms.radius; i = i + 1) {
          let z = lrnSub[index + i];
          sum = sum + z * z;
        }
        ${powOperatorSnippet}

        setOutputAtCoords(b, r, c, d, lrnSub[index] * powValue);
      }
    } `;
    return userCode;
  }
}
