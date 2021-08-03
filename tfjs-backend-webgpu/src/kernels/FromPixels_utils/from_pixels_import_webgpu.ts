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

import {FromPixelsProgram} from './from_pixels_webgpu';

export class FromPixelsImportProgram extends FromPixelsProgram {
  useWgsl = true;

  getUserCodeWgsl(): string {
    const userCode = `
    [[binding(2), group(0)]] var src: texture_external;

    [[stage(compute), workgroup_size(${this.workGroupSize[0]}, 1, 1)]]
    fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
      var flatIndexBase = i32(GlobalInvocationID.x) * uniforms.numChannels;
      var coords: vec3<u32> = getCoordsFromFlatIndex(u32(flatIndexBase));
      var texR: i32 = i32(coords[0]);
      var texC: i32 = i32(coords[1]);
      var depth: i32 = i32(coords[2]);
      var values = textureLoad(src, vec2<i32>(texC, texR));
      for (var i: i32 = 0; i < uniforms.numChannels; i = i + 1) {
        var value = values[i];
        var flatIndex = i32(flatIndexBase) + i;
        if (flatIndex < uniforms.size) {
          result.numbers[u32(flatIndex)] = i32(floor(255.0 * value));
        }
      }
    }
`;
    return userCode;
  }

  getImportTexture(device: GPUDevice, externalImage: HTMLVideoElement):
      GPUExternalTexture {
    const externalTextureDescriptor = {source: externalImage};
    return device.importExternalTexture(externalTextureDescriptor);
  }
}
