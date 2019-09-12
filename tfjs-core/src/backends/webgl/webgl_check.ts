/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

export function callAndCheck<T>(
    gl: WebGLRenderingContext, debugMode: boolean, func: () => T): T {
  const returnValue = func();
  if (debugMode) {
    checkWebGLError(gl);
  }
  return returnValue;
}

export function checkWebGLError(gl: WebGLRenderingContext) {
  const error = gl.getError();
  if (error !== gl.NO_ERROR) {
    throw new Error('WebGL Error: ' + getWebGLErrorMessage(gl, error));
  }
}

function getWebGLErrorMessage(
    gl: WebGLRenderingContext, status: number): string {
  switch (status) {
    case gl.NO_ERROR:
      return 'NO_ERROR';
    case gl.INVALID_ENUM:
      return 'INVALID_ENUM';
    case gl.INVALID_VALUE:
      return 'INVALID_VALUE';
    case gl.INVALID_OPERATION:
      return 'INVALID_OPERATION';
    case gl.INVALID_FRAMEBUFFER_OPERATION:
      return 'INVALID_FRAMEBUFFER_OPERATION';
    case gl.OUT_OF_MEMORY:
      return 'OUT_OF_MEMORY';
    case gl.CONTEXT_LOST_WEBGL:
      return 'CONTEXT_LOST_WEBGL';
    default:
      return `Unknown error code ${status}`;
  }
}
