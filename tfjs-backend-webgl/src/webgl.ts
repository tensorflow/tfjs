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

import {env} from '@tensorflow/tfjs-core';

import * as gpgpu_util from './gpgpu_util';
import * as webgl_util from './webgl_util';

export {MathBackendWebGL, WebGLMemoryInfo, WebGLTimingInfo} from './backend_webgl';
export {setWebGLContext} from './canvas_util';
export {GPGPUContext} from './gpgpu_context';
export {GPGPUProgram} from './gpgpu_math';
// WebGL specific utils.
export {gpgpu_util, webgl_util};

/**
 * Enforce use of half precision textures if available on the platform.
 *
 * @doc {heading: 'Environment', namespace: 'webgl'}
 */
export function forceHalfFloat(): void {
  env().set('WEBGL_FORCE_F16_TEXTURES', true);
}
