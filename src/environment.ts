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

import * as device_util from './device_util';
import {MathBackend} from './math/backends/backend';
import {NDArrayMath} from './math/math';
import * as util from './util';

export enum Type {
  NUMBER,
  BOOLEAN
}

export interface Features {
  // Whether the disjoint_query_timer extension is an available extension.
  'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED'?: boolean;
  // Whether the timer object from the disjoint_query_timer extension gives
  // timing information that is reliable.
  'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'?: boolean;
  // 0: No WebGL, 1: WebGL 1.0, 2: WebGL 2.0.
  'WEBGL_VERSION'?: number;
  // Whether writing & reading floating point textures is enabled. When
  // false, fall back to using unsigned byte textures.
  'WEBGL_FLOAT_TEXTURE_ENABLED'?: boolean;
  // Whether WEBGL_get_buffer_sub_data_async is enabled.
  'WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED'?: boolean;
}

export const URL_PROPERTIES: URLProperty[] = [
  {name: 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED', type: Type.BOOLEAN},
  {name: 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE', type: Type.BOOLEAN},
  {name: 'WEBGL_VERSION', type: Type.NUMBER},
  {name: 'WEBGL_FLOAT_TEXTURE_ENABLED', type: Type.BOOLEAN}, {
    name: 'WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED',
    type: Type.BOOLEAN
  }
];

export interface URLProperty {
  name: keyof Features;
  type: Type;
}

function getWebGLRenderingContext(webGLVersion: number): WebGLRenderingContext {
  if (webGLVersion === 0) {
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

function isWebGLVersionEnabled(webGLVersion: 1|2) {
  const gl = getWebGLRenderingContext(webGLVersion);
  if (gl != null) {
    loseContext(gl);
    return true;
  }
  return false;
}

function isWebGLDisjointQueryTimerEnabled(webGLVersion: number) {
  const gl = getWebGLRenderingContext(webGLVersion);

  const extensionName = webGLVersion === 1 ? 'EXT_disjoint_timer_query' :
                                             'EXT_disjoint_timer_query_webgl2';
  const ext = gl.getExtension(extensionName);
  const isExtEnabled = ext != null;
  if (gl != null) {
    loseContext(gl);
  }
  return isExtEnabled;
}

function isFloatTextureReadPixelsEnabled(webGLVersion: number): boolean {
  if (webGLVersion === 0) {
    return false;
  }

  const gl = getWebGLRenderingContext(webGLVersion);

  if (webGLVersion === 1) {
    if (gl.getExtension('OES_texture_float') == null) {
      return false;
    }
  } else {
    if (gl.getExtension('EXT_color_buffer_float') == null) {
      return false;
    }
  }

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

  const frameBufferComplete =
      (gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE);

  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, new Float32Array(4));

  const readPixelsNoError = gl.getError() === gl.NO_ERROR;

  loseContext(gl);

  return frameBufferComplete && readPixelsNoError;
}

function isWebGLGetBufferSubDataAsyncExtensionEnabled(webGLVersion: number) {
  if (webGLVersion !== 2) {
    return false;
  }
  const gl = getWebGLRenderingContext(webGLVersion);
  const ext = gl.getExtension('WEBGL_get_buffer_sub_data_async');
  const isEnabled = ext != null;
  loseContext(gl);
  return isEnabled;
}

export type BackendType = 'webgl'|'cpu';

export class Environment {
  private features: Features = {};
  private globalMath: NDArrayMath = null;
  // tslint:disable-next-line:no-any
  private backendRegistry: {[id in BackendType]: MathBackend} = {} as any;
  private prevBackendRegistry: {[id in BackendType]: MathBackend} = null;

  constructor(features?: Features) {
    if (features != null) {
      this.features = features;
    }
  }

  get<K extends keyof Features>(feature: K): Features[K] {
    if (feature in this.features) {
      return this.features[feature];
    }

    this.features[feature] = this.evaluateFeature(feature);

    return this.features[feature];
  }

  getBestBackend(): MathBackend {
    const orderedBackends: BackendType[] = ['webgl', 'cpu'];
    for (let i = 0; i < orderedBackends.length; ++i) {
      const backendId = orderedBackends[i];
      if (backendId in this.backendRegistry) {
        return this.backendRegistry[backendId];
      }
    }
    throw new Error('No backend found in registry.');
  }

  private evaluateFeature<K extends keyof Features>(feature: K): Features[K] {
    if (feature === 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED') {
      const webGLVersion = this.get('WEBGL_VERSION');

      if (webGLVersion === 0) {
        return false;
      }

      return isWebGLDisjointQueryTimerEnabled(webGLVersion);
    } else if (feature === 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') {
      return this.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED') &&
          !device_util.isMobile();
    } else if (feature === 'WEBGL_VERSION') {
      if (isWebGLVersionEnabled(2)) {
        return 2;
      } else if (isWebGLVersionEnabled(1)) {
        return 1;
      }
      return 0;
    } else if (feature === 'WEBGL_FLOAT_TEXTURE_ENABLED') {
      return isFloatTextureReadPixelsEnabled(this.get('WEBGL_VERSION'));
    } else if (
        feature === 'WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED') {
      return isWebGLGetBufferSubDataAsyncExtensionEnabled(
          this.get('WEBGL_VERSION'));
    }
    throw new Error(`Unknown feature ${feature}.`);
  }

  setFeatures(features: Features) {
    this.empty();
    this.features = features;
  }

  reset() {
    this.features = getFeaturesFromURL();
    if (this.globalMath != null) {
      this.globalMath.dispose();
      this.globalMath = null;
    }
    if (this.prevBackendRegistry != null) {
      for (const name in this.backendRegistry) {
        this.backendRegistry[name as BackendType].dispose();
      }
      this.backendRegistry = this.prevBackendRegistry;
      this.prevBackendRegistry = null;
    }
  }

  setMath(math: NDArrayMath) {
    this.globalMath = math;
  }

  getBackend(name: BackendType): MathBackend {
    return this.backendRegistry[name];
  }

  /**
   * Registers the backend to the global environment.
   *
   * @param factory: The backend factory function. When called, it should return
   *     an instance of the backend.
   * @return False if the creation/registration failed. True otherwise.
   */
  registerBackend(name: BackendType, factory: () => MathBackend): boolean {
    if (name in this.backendRegistry) {
      throw new Error(`${name} backend was already registered`);
    }
    try {
      const backend = factory();
      this.backendRegistry[name] = backend;
      return true;
    } catch (err) {
      return false;
    }
  }

  get math(): NDArrayMath {
    if (this.globalMath == null) {
      const bestBackend = this.getBestBackend();
      const safeMode = false;
      this.globalMath = new NDArrayMath(bestBackend, safeMode);
    }
    return this.globalMath;
  }

  private empty() {
    this.globalMath = null;
    this.prevBackendRegistry = this.backendRegistry;
    // tslint:disable-next-line:no-any
    this.backendRegistry = {} as any;
    this.features = null;
  }
}

// Expects flags from URL in the format ?dljsflags=FLAG1:1,FLAG2:true.
const DEEPLEARNJS_FLAGS_PREFIX = 'dljsflags';
function getFeaturesFromURL(): Features {
  const features: Features = {};

  if (typeof window === 'undefined') {
    return features;
  }

  const urlParams = util.getQueryParams(window.location.search);
  if (DEEPLEARNJS_FLAGS_PREFIX in urlParams) {
    const urlFlags: {[key: string]: string} = {};

    const keyValues = urlParams[DEEPLEARNJS_FLAGS_PREFIX].split(',');
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
        } else {
          console.warn(`Unknown URL param: ${urlProperty.name}.`);
        }
      }
    });
  }

  return features;
}

function getGlobalNamespace(): {ENV: Environment} {
  // tslint:disable-next-line:no-any
  let ns: any;
  if (typeof (window) !== 'undefined') {
    ns = window;
  } else if (typeof (global) !== 'undefined') {
    ns = global;
  } else {
    throw new Error('Could not find a global object');
  }
  return ns;
}

function getOrMakeEnvironment(): Environment {
  const ns = getGlobalNamespace();
  ns.ENV = ns.ENV || new Environment(getFeaturesFromURL());
  return ns.ENV;
}

export let ENV = getOrMakeEnvironment();
