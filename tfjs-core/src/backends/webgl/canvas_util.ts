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

const WEBGL_ATTRIBUTES: WebGLContextAttributes = {
  alpha: false,
  antialias: false,
  premultipliedAlpha: false,
  preserveDrawingBuffer: false,
  depth: false,
  stencil: false,
  failIfMajorPerformanceCaveat: true
};

export function createCanvas(webGLVersion: number) {
  if (typeof OffscreenCanvas !== 'undefined' && webGLVersion === 2) {
    return new OffscreenCanvas(300, 150);
  } else if (typeof document !== 'undefined') {
    return document.createElement('canvas');
  } else {
    throw new Error('Cannot create a canvas in this context');
  }
}

export function browserContextCleanup(context: WebGLRenderingContext) {
  if (context == null) {
    throw new Error('Unexpected exception: context to be cleaned up is null!');
  }
  const canvas = context.canvas;
  if (canvas != null && canvas.remove != null) {
    canvas.remove();
  }
}

export function browserContextFactory(webGLVersion: number):
    WebGLRenderingContext {
  if (webGLVersion !== 1 && webGLVersion !== 2) {
    throw new Error('Cannot get WebGL rendering context, WebGL is disabled.');
  }
  const canvas = createCanvas(webGLVersion);

  canvas.addEventListener('webglcontextlost', (ev: Event) => {
    ev.preventDefault();
  }, false);
  if (webGLVersion === 1) {
    return (canvas.getContext('webgl', WEBGL_ATTRIBUTES) ||
            canvas.getContext('experimental-webgl', WEBGL_ATTRIBUTES)) as
        WebGLRenderingContext;
  }
  return canvas.getContext('webgl2', WEBGL_ATTRIBUTES) as WebGLRenderingContext;
}
