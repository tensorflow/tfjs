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

import * as tf from '@tensorflow/tfjs-core';
import {FromCameraProgram} from './from_camera_program';

interface FromCameraInputs {}

export interface FromCameraAttrs {
  texture: WebGLTexture;
  width: number;
  height: number;
  numChannels: number;
}

console.log('registering fromCamera kernel');
tf.registerKernel({
  kernelName: 'FromCamera',
  backendName: 'rn-webgl',
  kernelFunc: fromCamera as {} as tf.KernelFunc,
});

export function fromCamera(args: {
  inputs: FromCameraInputs,
  backend: tf.webgl.MathBackendWebGL,
  attrs: FromCameraAttrs
}): tf.TensorInfo {
  const {backend, attrs} = args;
  const outShape = [attrs.height, attrs.width, attrs.numChannels];

  const program = new FromCameraProgram(outShape);
  console.log(
      'before compile and run', attrs.width, attrs.height, attrs.texture);
  const customSetup = program.getCustomSetupFunc(attrs.texture);
  const r = backend.compileAndRun(program, [], 'int32', customSetup);
  // console.log('runWebGLProgram');
  // const res = backend.runWebGLProgram(program, [], 'int32', customSetup);

  return r;
}
