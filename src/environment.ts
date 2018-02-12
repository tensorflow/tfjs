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
import {doc} from './doc';
import {Engine, MemoryInfo} from './engine';
import {KernelBackend} from './kernels/backend';
import {NDArrayMath} from './math';
import * as util from './util';

export enum Type {
  NUMBER,
  BOOLEAN,
  STRING
}

export interface Features {
  // Whether to enable debug mode.
  'DEBUG'?: boolean;
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
  // Whether writing & reading floating point textures is enabled. When
  // false, fall back to using unsigned byte textures.
  'WEBGL_FLOAT_TEXTURE_ENABLED'?: boolean;
  // Whether WEBGL_get_buffer_sub_data_async is enabled.
  'WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED'?: boolean;
  'BACKEND'?: BackendType;
}

export const URL_PROPERTIES: URLProperty[] = [
  {name: 'DEBUG', type: Type.BOOLEAN},
  {name: 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION', type: Type.NUMBER},
  {name: 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE', type: Type.BOOLEAN},
  {name: 'WEBGL_VERSION', type: Type.NUMBER},
  {name: 'WEBGL_FLOAT_TEXTURE_ENABLED', type: Type.BOOLEAN}, {
    name: 'WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED',
    type: Type.BOOLEAN
  },
  {name: 'BACKEND', type: Type.STRING}
];

export interface URLProperty {
  name: keyof Features;
  type: Type;
}

function hasExtension(gl: WebGLRenderingContext, extensionName: string) {
  const ext = gl.getExtension(extensionName);
  return ext != null;
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

function getWebGLDisjointQueryTimerVersion(webGLVersion: number): number {
  if (webGLVersion === 0) {
    return 0;
  }

  let queryTimerVersion: number;
  const gl = getWebGLRenderingContext(webGLVersion);

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

function isFloatTextureReadPixelsEnabled(webGLVersion: number): boolean {
  if (webGLVersion === 0) {
    return false;
  }

  const gl = getWebGLRenderingContext(webGLVersion);

  if (webGLVersion === 1) {
    if (!hasExtension(gl, 'OES_texture_float')) {
      return false;
    }
  } else {
    if (!hasExtension(gl, 'EXT_color_buffer_float')) {
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

  const isEnabled = hasExtension(gl, 'WEBGL_get_buffer_sub_data_async');
  loseContext(gl);
  return isEnabled;
}

/** @docalias 'webgl'|'cpu' */
export type BackendType = 'webgl'|'cpu';

/** List of currently supported backends ordered by preference. */
const SUPPORTED_BACKENDS: BackendType[] = ['webgl', 'cpu'];

export class Environment {
  private features: Features = {};
  private globalMath: NDArrayMath;
  private globalEngine: Engine;
  private BACKEND_REGISTRY: {[id: string]: KernelBackend} = {};
  private backends: {[id: string]: KernelBackend} = this.BACKEND_REGISTRY;
  private currentBackendType: BackendType;

  constructor(features?: Features) {
    if (features != null) {
      this.features = features;
    }

    if (this.get('DEBUG')) {
      console.warn(
          'Debugging mode is ON. The output of every math call will ' +
          'be downloaded to CPU and checked for NaNs. ' +
          'This significantly impacts performance.');
    }
  }

  /**
   * Sets the backend (cpu, webgl, etc) responsible for creating tensors and
   * executing operations on those tensors.
   *
   * @param backendType The backend type. Currently supports 'webgl'|'cpu'.
   * @param safeMode Defaults to false. In safe mode, you are forced to
   *     construct tensors and call math operations inside a dl.tidy() which
   *     will automatically clean up intermediate tensors.
   */
  @doc({heading: 'Environment'})
  static setBackend(backendType: BackendType, safeMode = false) {
    if (!(backendType in ENV.backends)) {
      throw new Error(`Backend type '${backendType}' not found in registry`);
    }
    ENV.globalMath = new NDArrayMath(backendType, safeMode);
  }

  /**
   * Returns the current backend (cpu, webgl, etc). The backend is responsible
   * for creating tensors and executing operations on those tensors.
   */
  @doc({heading: 'Environment'})
  static getBackend(): BackendType {
    ENV.initEngine();
    return ENV.currentBackendType;
  }

  /**
   * Returns memory info at the current time in the program. The result is an
   * object with the following properties:
   *
   * - `numBytes`: number of bytes allocated (undisposed) at this time.
   * - `numTensors`: number of unique tensors allocated
   * - `numDataBuffers`: number of unique data buffers allocated
   *   (undisposed) at this time, which is ≤ the number of tensors
   *   (e.g. `a.reshape(newShape)` makes a new Tensor that shares the same
   *   data buffer with `a`).
   * - `unreliable`: optional boolean:
   *    - On WebGL, not present (always reliable).
   *    - On CPU, true. Due to automatic garbage collection, these numbers
   *     represent undisposed tensors, i.e. not wrapped in `dl.tidy()`, or
   *     lacking a call to `tensor.dispose()`.
   * - `backendInfo`: Backend-specific information.
   */
  @doc({heading: 'Performance', subheading: 'Memory'})
  static memory(): MemoryInfo {
    return ENV.engine.memory();
  }

  get<K extends keyof Features>(feature: K): Features[K] {
    if (feature in this.features) {
      return this.features[feature];
    }

    this.features[feature] = this.evaluateFeature(feature);

    return this.features[feature];
  }

  set<K extends keyof Features>(feature: K, value: Features[K]): void {
    this.features[feature] = value;
  }

  getBestBackendType(): BackendType {
    for (let i = 0; i < SUPPORTED_BACKENDS.length; ++i) {
      const backendId = SUPPORTED_BACKENDS[i];
      if (backendId in this.backends) {
        return backendId;
      }
    }
    throw new Error('No backend found in registry.');
  }

  private evaluateFeature<K extends keyof Features>(feature: K): Features[K] {
    if (feature === 'DEBUG') {
      return false;
    } else if (feature === 'BACKEND') {
      return this.getBestBackendType();
    } else if (feature === 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') {
      const webGLVersion = this.get('WEBGL_VERSION');

      if (webGLVersion === 0) {
        return 0;
      }

      return getWebGLDisjointQueryTimerVersion(webGLVersion);
    } else if (feature === 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') {
      return this.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0 &&
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
    this.reset();
    this.features = features;
    this.backends = {};
  }

  reset() {
    this.features = getFeaturesFromURL();
    if (this.globalMath != null) {
      this.globalMath.dispose();
      this.globalMath = null;
      this.globalEngine = null;
    }
    if (this.backends !== this.BACKEND_REGISTRY) {
      for (const name in this.backends) {
        this.backends[name].dispose();
      }
      this.backends = this.BACKEND_REGISTRY;
    }
  }

  setMath(
      math: NDArrayMath, backend?: BackendType|KernelBackend,
      safeMode = false) {
    if (this.globalMath === math) {
      return;
    }
    let customBackend = false;
    if (typeof backend === 'string') {
      this.currentBackendType = backend;
      backend = ENV.findBackend(backend);
    } else {
      customBackend = true;
      this.currentBackendType = 'custom' as BackendType;
    }
    this.globalEngine = new Engine(backend, customBackend, safeMode);
    this.globalMath = math;
  }

  findBackend(name: BackendType): KernelBackend {
    return this.backends[name];
  }

  /**
   * Adds a custom backend. Usually used in tests to simulate different
   * environments.
   *
   * @param factory: The backend factory function. When called, it should return
   *     an instance of the backend.
   * @return False if the creation/registration failed. True otherwise.
   */
  addCustomBackend(name: BackendType, factory: () => KernelBackend): boolean {
    if (name in this.backends) {
      throw new Error(`${name} backend was already registered`);
    }
    try {
      const backend = factory();
      this.backends[name] = backend;
      return true;
    } catch (err) {
      return false;
    }
  }

  /**
   * Registers a global backend. The registration should happen when importing
   * a module file (e.g. when importing `backend_webgl.ts`), and is used for
   * modular builds (e.g. custom deeplearn.js bundle with only webgl support).
   *
   * @param factory: The backend factory function. When called, it should
   * return an instance of the backend.
   * @return False if the creation/registration failed. True otherwise.
   */
  registerBackend(name: BackendType, factory: () => KernelBackend): boolean {
    if (name in this.BACKEND_REGISTRY) {
      throw new Error(`${name} backend was already registered as global`);
    }
    try {
      const backend = factory();
      this.BACKEND_REGISTRY[name] = backend;
      return true;
    } catch (err) {
      return false;
    }
  }

  /** @deprecated. Use ENV.engine. */
  get math(): NDArrayMath {
    if (this.globalEngine == null) {
      this.initEngine();
    }
    return this.globalMath;
  }

  get engine(): Engine {
    if (this.globalEngine == null) {
      this.initEngine();
    }
    return this.globalEngine;
  }

  private initEngine() {
    this.globalMath = new NDArrayMath(ENV.get('BACKEND'), false /* safeMode */);
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
