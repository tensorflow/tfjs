/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

export interface Features {
  // Whether to enable debug mode.
  'DEBUG'?: boolean;
  // Whether we are in a browser (as versus, say, node.js) environment.
  'IS_BROWSER'?: boolean;
  // Whether we are in the Node.js environment.
  'IS_NODE'?: boolean;
  // The disjoint_query_timer extension version.
  // 0: disabled, 1: EXT_disjoint_timer_query, 2:
  // EXT_disjoint_timer_query_webgl2.
  // In Firefox with WebGL 2.0,
  // EXT_disjoint_timer_query_webgl2 is not available, so we must use the
  // WebGL 1.0 extension.
  'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION'?: number;
  // Whether the timer object from the disjoint_query_timer extension gives
  // timing information that is reliable.
  'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'?: boolean;
  // 0: No WebGL, 1: WebGL 1.0, 2: WebGL 2.0.
  'WEBGL_VERSION'?: number;
  // True if WebGL is supported.
  'HAS_WEBGL'?: boolean;
  // Whether rendering to float32 textures is enabled. If disabled, renders to
  // float16 textures.
  'WEBGL_RENDER_FLOAT32_ENABLED'?: boolean;
  // Whether downloading float textures is enabled. If disabled, uses IEEE 754
  // encoding of the float32 values to 4 uint8 when downloading.
  'WEBGL_DOWNLOAD_FLOAT_ENABLED'?: boolean;
  // Whether the fence API is available.
  'WEBGL_FENCE_API_ENABLED'?: boolean;
  'BACKEND'?: string;
  // Test precision for unit tests. This is decreased when we can't render
  // float32 textures.
  'TEST_EPSILON'?: number;
  'IS_CHROME'?: boolean;
  // True if running unit tests.
  'IS_TEST'?: boolean;
}

export enum Type {
  NUMBER,
  BOOLEAN,
  STRING
}

export const URL_PROPERTIES: URLProperty[] = [
  {name: 'DEBUG', type: Type.BOOLEAN}, {name: 'IS_BROWSER', type: Type.BOOLEAN},
  {name: 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION', type: Type.NUMBER},
  {name: 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE', type: Type.BOOLEAN},
  {name: 'WEBGL_VERSION', type: Type.NUMBER},
  {name: 'WEBGL_RENDER_FLOAT32_ENABLED', type: Type.BOOLEAN},
  {name: 'WEBGL_DOWNLOAD_FLOAT_ENABLED', type: Type.BOOLEAN},
  {name: 'WEBGL_FENCE_API_ENABLED', type: Type.BOOLEAN},
  {name: 'BACKEND', type: Type.STRING}
];

export interface URLProperty {
  name: keyof Features;
  type: Type;
}

export function isWebGLVersionEnabled(webGLVersion: 1|2, isBrowser: boolean) {
  let gl;
  try {
    gl = getWebGLRenderingContext(webGLVersion, isBrowser);
  } catch (e) {
    return false;
  }

  if (gl != null) {
    loseContext(gl);
    return true;
  }
  return false;
}

export function getWebGLDisjointQueryTimerVersion(
    webGLVersion: number, isBrowser: boolean): number {
  if (webGLVersion === 0) {
    return 0;
  }

  let queryTimerVersion: number;
  const gl = getWebGLRenderingContext(webGLVersion, isBrowser);

  if (hasExtension(gl, 'EXT_disjoint_timer_query_webgl2') &&
      webGLVersion === 2) {
    queryTimerVersion = 2;
  } else if (hasExtension(gl, 'EXT_disjoint_timer_query')) {
    queryTimerVersion = 1;
  } else {
    queryTimerVersion = 0;
  }

  if (gl != null) {
    loseContext(gl);
  }
  return queryTimerVersion;
}

export function isRenderToFloatTextureEnabled(
    webGLVersion: number, isBrowser: boolean): boolean {
  if (webGLVersion === 0) {
    return false;
  }

  const gl = getWebGLRenderingContext(webGLVersion, isBrowser);

  if (webGLVersion === 1) {
    if (!hasExtension(gl, 'OES_texture_float')) {
      return false;
    }
  } else {
    if (!hasExtension(gl, 'EXT_color_buffer_float')) {
      return false;
    }
  }

  createFloatTextureAndBindToFramebuffer(gl, webGLVersion);

  const isFrameBufferComplete =
      gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;

  loseContext(gl);
  return isFrameBufferComplete;
}

export function isDownloadFloatTextureEnabled(
    webGLVersion: number, isBrowser: boolean): boolean {
  if (webGLVersion === 0) {
    return false;
  }

  const gl = getWebGLRenderingContext(webGLVersion, isBrowser);

  if (webGLVersion === 1) {
    if (!hasExtension(gl, 'OES_texture_float')) {
      return false;
    }
  } else {
    if (!hasExtension(gl, 'EXT_color_buffer_float')) {
      return false;
    }
  }

  createFloatTextureAndBindToFramebuffer(gl, webGLVersion);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, new Float32Array(4));

  const readPixelsNoError = gl.getError() === gl.NO_ERROR;

  loseContext(gl);

  return readPixelsNoError;
}

export function isWebGLFenceEnabled(webGLVersion: number, isBrowser: boolean) {
  if (webGLVersion !== 2) {
    return false;
  }
  const gl = getWebGLRenderingContext(webGLVersion, isBrowser);

  // tslint:disable-next-line:no-any
  const isEnabled = (gl as any).fenceSync != null;
  loseContext(gl);
  return isEnabled;
}

export function isChrome() {
  return typeof navigator !== 'undefined' && navigator != null &&
      navigator.userAgent != null && /Chrome/.test(navigator.userAgent) &&
      /Google Inc/.test(navigator.vendor);
}

// Expects flags from URL in the format ?tfjsflags=FLAG1:1,FLAG2:true.
const TENSORFLOWJS_FLAGS_PREFIX = 'tfjsflags';
export function getFeaturesFromURL(): Features {
  const features: Features = {};

  if (typeof window === 'undefined' || typeof window.location === 'undefined') {
    return features;
  }

  const urlParams = getQueryParams(window.location.search);
  if (TENSORFLOWJS_FLAGS_PREFIX in urlParams) {
    const urlFlags: {[key: string]: string} = {};

    const keyValues = urlParams[TENSORFLOWJS_FLAGS_PREFIX].split(',');
    keyValues.forEach(keyValue => {
      const [key, value] = keyValue.split(':') as [string, string];
      urlFlags[key] = value;
    });

    URL_PROPERTIES.forEach(urlProperty => {
      if (urlProperty.name in urlFlags) {
        console.log(
            `Setting feature override from URL ${urlProperty.name}: ` +
            `${urlFlags[urlProperty.name]}`);
        if (urlProperty.type === Type.NUMBER) {
          features[urlProperty.name] = +urlFlags[urlProperty.name];
        } else if (urlProperty.type === Type.BOOLEAN) {
          features[urlProperty.name] = urlFlags[urlProperty.name] === 'true';
        } else if (urlProperty.type === Type.STRING) {
          // tslint:disable-next-line:no-any
          features[urlProperty.name] = urlFlags[urlProperty.name] as any;
        } else {
          console.warn(`Unknown URL param: ${urlProperty.name}.`);
        }
      }
    });
  }

  return features;
}

function hasExtension(gl: WebGLRenderingContext, extensionName: string) {
  const ext = gl.getExtension(extensionName);
  return ext != null;
}

function getWebGLRenderingContext(
    webGLVersion: number, isBrowser: boolean): WebGLRenderingContext {
  if (webGLVersion === 0 || !isBrowser) {
    throw new Error('Cannot get WebGL rendering context, WebGL is disabled.');
  }

  const tempCanvas = document.createElement('canvas');

  if (webGLVersion === 1) {
    return (tempCanvas.getContext('webgl') ||
            tempCanvas.getContext('experimental-webgl')) as
        WebGLRenderingContext;
  }
  return tempCanvas.getContext('webgl2') as WebGLRenderingContext;
}

function loseContext(gl: WebGLRenderingContext) {
  if (gl != null) {
    const loseContextExtension = gl.getExtension('WEBGL_lose_context');
    if (loseContextExtension == null) {
      throw new Error(
          'Extension WEBGL_lose_context not supported on this browser.');
    }
    loseContextExtension.loseContext();
  }
}

function createFloatTextureAndBindToFramebuffer(
    gl: WebGLRenderingContext, webGLVersion: number) {
  const frameBuffer = gl.createFramebuffer();
  const texture = gl.createTexture();

  gl.bindTexture(gl.TEXTURE_2D, texture);

  // tslint:disable-next-line:no-any
  const internalFormat = webGLVersion === 2 ? (gl as any).RGBA32F : gl.RGBA;
  gl.texImage2D(
      gl.TEXTURE_2D, 0, internalFormat, 1, 1, 0, gl.RGBA, gl.FLOAT, null);

  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
}

export function getQueryParams(queryString: string): {[key: string]: string} {
  const params = {};
  queryString.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g, (s, ...t) => {
    decodeParam(params, t[0], t[1]);
    return t.join('=');
  });
  return params;
}

function decodeParam(
    params: {[key: string]: string}, name: string, value?: string) {
  params[decodeURIComponent(name)] = decodeURIComponent(value || '');
}
