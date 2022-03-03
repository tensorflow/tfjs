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

import {WebGPUProgram} from './webgpu_program';
import {computeDispatch} from './webgpu_util';

export class nhwcTophwc4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  uniforms = 'sizes_';
  dispatch: [number, number, number];
  variableNames = ['x'];
  workGroupSize: [number, number, number] = [4, 4, 4];
  fullShader = true;

  constructor(outputShape: number[]) {
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0, 3]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize, [1, 1, 4]);
    this.shaderKey = 'nhwcTophwc4';
  }

  getUserCode(): string {
    const userCode = `
  struct Matrix0 {
    elements: array<vec4<f32>>;
  };

  struct Matrix1 {
    elements: array<f32>;
  };

  struct Uniforms {
    sizes_ : vec4<i32>; // w, h, slice, ch
  }

  @group(0) @binding(0) var<storage, write> output_data: Matrix0;
  @group(0) @binding(1) var<storage, read> input_data: Matrix1;
  @group(0) @binding(2) var<uniform> uniforms : Uniforms;

    @stage(compute) @workgroup_size(4, 4, 4)
    fn main(@builtin(local_invocation_id) localId : vec3<u32>,
            @builtin(global_invocation_id) globalId : vec3<u32>) {
      let gid = vec3<i32>(globalId);
      if (gid.x >= uniforms.sizes_.x || gid.y >= uniforms.sizes_.y || gid.z >= uniforms.sizes_.z) {
        return;
      }
      var v = vec4<f32>(0.0);
      var dst_channel = gid.z * 4;
      var index = (gid.y * uniforms.sizes_.x + gid.x) * uniforms.sizes_.w + dst_channel;
      for (var i = 0; i < 4; i = i + 1) {
        if (dst_channel >= uniforms.sizes_.w)
        {
          break;
        }
        v[i] = input_data.elements[index];
        index = index + 1;
        dst_channel = dst_channel + 1;
      }
      output_data.elements[(gid.z * uniforms.sizes_.y + gid.y) * uniforms.sizes_.x + gid.x] = v;
    }
     `;
    return userCode;
  }
}
