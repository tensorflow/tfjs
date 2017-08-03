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

import {GPGPUContext} from './gpgpu_context';
import * as pool_gpu from './pool_gpu';

export function getFragmentShaderMinPoolSource(
    xShapeRCD: [number, number, number], fSize: number, stride: number,
    pad: number) {
  return pool_gpu.getFragmentShaderPoolCommonSource(
      xShapeRCD, fSize, stride, pad, 'min', false);
}

export function minPool(
    gpgpu: GPGPUContext, program: WebGLProgram, x: WebGLTexture,
    result: WebGLTexture, resultShapeRowCol: [number, number]) {
  pool_gpu.poolCommon(gpgpu, program, x, result, resultShapeRowCol);
}