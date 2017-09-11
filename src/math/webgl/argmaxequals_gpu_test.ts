/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {ArgMaxEqualsProgram} from './argmaxequals_gpu';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {TextureManager} from './texture_manager';
import {Array2D, Scalar, initializeGPU} from '../ndarray';

function uploadArgMaxEqualsDownload(
    a: Float32Array, b: Float32Array, rows: number, columns: number): number {
  const aArr = Array2D.new([rows, columns], a);
  const bArr = Array2D.new([rows, columns], b);
  const gpgpu = new GPGPUContext();
  const textureManager = new TextureManager(gpgpu);
  initializeGPU(gpgpu, textureManager);
  const out = Scalar.new(0);
  const program = new ArgMaxEqualsProgram(aArr.size, bArr.size);
  const binary = gpgpu_math.compileProgram(gpgpu, program, [aArr, bArr], out);
  gpgpu_math.runProgram(binary, [aArr, bArr], out);
  const result = out.get();
  aArr.dispose();
  bArr.dispose();
  textureManager.dispose();
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();
  return result;
}

describe('argmaxequals_gpu ArgMin', () => {
  it('one value in each array', () => {
    const a = new Float32Array([3]);
    const b = new Float32Array([3]);
    const equals = uploadArgMaxEqualsDownload(a, b, 1, 1);
    expect(equals).toEqual(1);
  });

  it('different argmax values', () => {
    const a = new Float32Array([2, 3]);
    const b = new Float32Array([3, 2]);
    const equals = uploadArgMaxEqualsDownload(a, b, 1, 2);
    expect(equals).toEqual(0);
  });

  it('same argmax values', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 4, 3, 2, 1]);
    const b = new Float32Array([10, 2, 30, 4, 50, 4, 30, 2, 10]);
    const equals = uploadArgMaxEqualsDownload(a, b, 1, 9);
    expect(equals).toEqual(1);
  });
});
