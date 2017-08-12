/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import {NDArray, initializeGPU, Array2D} from '../ndarray';
import {UnaryOp, UnaryOpProgram} from './unaryop_gpu';
import {TextureManager} from './texture_manager';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';

export function uploadUnaryDownload(a: NDArray, op: UnaryOp): Float32Array {
  const gpgpu = new GPGPUContext();
  const textureManager = new TextureManager(gpgpu);
  initializeGPU(gpgpu, textureManager);
  const out = Array2D.zerosLike(a);
  const program = new UnaryOpProgram(a.shape, op);
  const binary = gpgpu_math.compileProgram(gpgpu, program, [a], out);
  gpgpu_math.runProgram(binary, [a], out);
  const result = out.getValues();
  textureManager.dispose();
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();
  return result;
}
