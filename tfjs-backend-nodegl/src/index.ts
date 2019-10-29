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
// TODO(kreeger): Fix binding definition in node-gles before switching to a
// non-require import style.
// tslint:disable-next-line:no-require-imports
const nodeGles = require('node-gles');

const nodeGl = nodeGles.binding.createWebGLRenderingContext();

// TODO(kreeger): These are hard-coded GL integration flags. These need to be
// updated to ensure they work on all systems with proper exception reporting.
tf.env().set('WEBGL_VERSION', 2);
tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);
tf.env().set('WEBGL_DOWNLOAD_FLOAT_ENABLED', true);
tf.env().set('WEBGL_FENCE_API_ENABLED', true);  // OpenGL ES 3.0 and higher..
tf.env().set(
    'WEBGL_MAX_TEXTURE_SIZE', nodeGl.getParameter(nodeGl.MAX_TEXTURE_SIZE));
tf.webgl.setWebGLContext(2, nodeGl);

tf.registerBackend('headless-nodegl', () => {
  // TODO(kreeger): Consider moving all GL creation here. However, weak-ref to
  // GL context tends to cause an issue when running unit tests:
  // https://github.com/tensorflow/tfjs/issues/1732
  return new tf.webgl.MathBackendWebGL(new tf.webgl.GPGPUContext(nodeGl));
}, 3 /* priority */);
