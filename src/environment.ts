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
import * as util from './util';

export enum Type {
  NUMBER,
  BOOLEAN,
  STRING
}

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
  // Whether rendering to float32 textures is enabled. If disabled, renders to
  // float16 textures.
  'WEBGL_RENDER_FLOAT32_ENABLED'?: boolean;
  // Whether downloading float textures is enabled. If disabled, uses IEEE 754
  // encoding of the float32 values to 4 uint8 when downloading.
  'WEBGL_DOWNLOAD_FLOAT_ENABLED'?: boolean;
  // Whether WEBGL_get_buffer_sub_data_async is enabled.
  'WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED'?: boolean;
  'BACKEND'?: string;
  // Test precision for unit tests. This is decreased when we can't render
  // float32 textures.
  'TEST_EPSILON'?: number;
  'IS_CHROME'?: boolean;
}

export const URL_PROPERTIES: URLProperty[] = [
  {name: 'DEBUG', type: Type.BOOLEAN}, {name: 'IS_BROWSER', type: Type.BOOLEAN},
  {name: 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION', type: Type.NUMBER},
  {name: 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE', type: Type.BOOLEAN},
  {name: 'WEBGL_VERSION', type: Type.NUMBER},
  {name: 'WEBGL_RENDER_FLOAT32_ENABLED', type: Type.BOOLEAN},
  {name: 'WEBGL_DOWNLOAD_FLOAT_ENABLED', type: Type.BOOLEAN}, {
    name: 'WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED',
    type: Type.BOOLEAN
  },
  {name: 'BACKEND', type: Type.STRING}
];

export interface URLProperty {
  name: keyof Features;
  type: Type;
}

const TEST_EPSILON_FLOAT32_ENABLED = 1e-3;
const TEST_EPSILON_FLOAT32_DISABLED = 1e-1;

function hasExtension(gl: WebGLRenderingContext, extensionName: string) {
  const ext = gl.getExtension(extensionName);
  return ext != null;
}

function getWebGLRenderingContext(webGLVersion: number): WebGLRenderingContext {
  if (webGLVersion === 0 || !ENV.get('IS_BROWSER')) {
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
  let gl;
  try {
    gl = getWebGLRenderingContext(webGLVersion);
  } catch (e) {
    return false;
  }

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

function isRenderToFloatTextureEnabled(webGLVersion: number): boolean {
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

  createFloatTextureAndBindToFramebuffer(gl, webGLVersion);

  const isFrameBufferComplete =
      gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;

  loseContext(gl);
  return isFrameBufferComplete;
}

function isDownloadFloatTextureEnabled(webGLVersion: number): boolean {
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

  createFloatTextureAndBindToFramebuffer(gl, webGLVersion);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, new Float32Array(4));

  const readPixelsNoError = gl.getError() === gl.NO_ERROR;

  loseContext(gl);

  return readPixelsNoError;
}

function isWebGLGetBufferSubDataAsyncExtensionEnabled(webGLVersion: number) {
  // TODO(nsthorat): Remove this once we fix
  // https://github.com/tensorflow/tfjs/issues/137
  if (webGLVersion > 0) {
    return false;
  }

  if (webGLVersion !== 2) {
    return false;
  }
  const gl = getWebGLRenderingContext(webGLVersion);

  const isEnabled = hasExtension(gl, 'WEBGL_get_buffer_sub_data_async');
  loseContext(gl);
  return isEnabled;
}

function isChrome() {
  return navigator != null && navigator.userAgent != null &&
      /Chrome/.test(navigator.userAgent) && /Google Inc/.test(navigator.vendor);
}

export class Environment {
  private features: Features = {};
  private globalEngine: Engine;
  private registry:
      {[id: string]: {backend: KernelBackend, priority: number}} = {};
  private currentBackend: string;

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
   * Note this disposes the current backend, if any, as well as any tensors
   * associated with it.  A new backend is initialized, even if it is of the
   * same type as the previous one.
   *
   * @param backendType The backend type. Currently supports `'webgl'|'cpu'` in
   *     the browser, and `'tensorflow'` under node.js (requires tfjs-node).
   * @param safeMode Defaults to false. In safe mode, you are forced to
   *     construct tensors and call math operations inside a `tidy()` which
   *     will automatically clean up intermediate tensors.
   */
  @doc({heading: 'Environment'})
  static setBackend(backendType: string, safeMode = false) {
    if (!(backendType in ENV.registry)) {
      throw new Error(`Backend type '${backendType}' not found in registry`);
    }
    ENV.initBackend(backendType, safeMode);
  }

  /**
   * Returns the current backend (cpu, webgl, etc). The backend is responsible
   * for creating tensors and executing operations on those tensors.
   */
  @doc({heading: 'Environment'})
  static getBackend(): string {
    ENV.initDefaultBackend();
    return ENV.currentBackend;
  }

  /**
   * Dispose all variables kept in backend engine.
   */
  @doc({heading: 'Environment'})
  static disposeVariables(): void {
    ENV.engine.disposeVariables();
  }

  /**
   * Returns memory info at the current time in the program. The result is an
   * object with the following properties:
   *
   * - `numBytes`: Number of bytes allocated (undisposed) at this time.
   * - `numTensors`: Number of unique tensors allocated.
   * - `numDataBuffers`: Number of unique data buffers allocated
   *   (undisposed) at this time, which is ≤ the number of tensors
   *   (e.g. `a.reshape(newShape)` makes a new Tensor that shares the same
   *   data buffer with `a`).
   * - `unreliable`: `Optional` `boolean`:
   *    - On WebGL, not present (always reliable).
   *    - On CPU, true. Due to automatic garbage collection, these numbers
   *     represent undisposed tensors, i.e. not wrapped in `tidy()`, or
   *     lacking a call to `tensor.dispose()`.
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

  getFeatures(): Features {
    return this.features;
  }

  set<K extends keyof Features>(feature: K, value: Features[K]): void {
    this.features[feature] = value;
  }

  getBestBackendType(): string {
    if (Object.keys(this.registry).length === 0) {
      throw new Error('No backend found in registry.');
    }
    const sortedBackends = Object.keys(this.registry)
                               .map(name => {
                                 return {name, entry: this.registry[name]};
                               })
                               .sort((a, b) => {
                                 // Highest priority comes first.
                                 return b.entry.priority - a.entry.priority;
                               });
    return sortedBackends[0].name;
  }

  private evaluateFeature<K extends keyof Features>(feature: K): Features[K] {
    if (feature === 'DEBUG') {
      return false;
    } else if (feature === 'IS_BROWSER') {
      return typeof window !== 'undefined';
    } else if (feature === 'IS_NODE') {
      return (typeof process !== 'undefined') &&
          (typeof process.versions.node !== 'undefined');
    } else if (feature === 'IS_CHROME') {
      return isChrome();
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
    } else if (feature === 'WEBGL_RENDER_FLOAT32_ENABLED') {
      return isRenderToFloatTextureEnabled(this.get('WEBGL_VERSION'));
    } else if (feature === 'WEBGL_DOWNLOAD_FLOAT_ENABLED') {
      return isDownloadFloatTextureEnabled(this.get('WEBGL_VERSION'));
    } else if (
        feature === 'WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED') {
      return isWebGLGetBufferSubDataAsyncExtensionEnabled(
          this.get('WEBGL_VERSION'));
    } else if (feature === 'TEST_EPSILON') {
      if (this.get('WEBGL_RENDER_FLOAT32_ENABLED')) {
        return TEST_EPSILON_FLOAT32_ENABLED;
      }
      return TEST_EPSILON_FLOAT32_DISABLED;
    }
    throw new Error(`Unknown feature ${feature}.`);
  }

  setFeatures(features: Features) {
    this.features = features;
  }

  reset() {
    this.features = getFeaturesFromURL();
    if (this.globalEngine != null) {
      this.globalEngine = null;
    }
  }

  private initBackend(backendType?: string, safeMode = false) {
    this.currentBackend = backendType;
    const backend = ENV.findBackend(backendType);
    this.globalEngine = new Engine(backend, safeMode);
  }

  findBackend(name: string): KernelBackend {
    if (!(name in this.registry)) {
      return null;
    }
    return this.registry[name].backend;
  }

  /**
   * Registers a global backend. The registration should happen when importing
   * a module file (e.g. when importing `backend_webgl.ts`), and is used for
   * modular builds (e.g. custom tfjs bundle with only webgl support).
   *
   * @param factory: The backend factory function. When called, it should
   * return an instance of the backend.
   * @param priority The priority of the backend (higher = more important).
   *     In case multiple backends are registered, `getBestBackendType` uses
   *     priority to find the best backend. Defaults to 1.
   * @return False if the creation/registration failed. True otherwise.
   */
  registerBackend(name: string, factory: () => KernelBackend, priority = 1):
      boolean {
    if (name in this.registry) {
      console.warn(`${name} backend was already registered`);
    }
    try {
      const backend = factory();
      this.registry[name] = {backend, priority};
      return true;
    } catch (err) {
      console.warn(`Registration of backend ${name} failed`);
      console.warn(err.stack || err.message);
      return false;
    }
  }

  removeBackend(name: string): void {
    if (!(name in this.registry)) {
      throw new Error(`${name} backend not found in registry`);
    }
    this.registry[name].backend.dispose();
    delete this.registry[name];
  }

  get engine(): Engine {
    this.initDefaultBackend();
    return this.globalEngine;
  }

  private initDefaultBackend() {
    if (this.globalEngine == null) {
      this.initBackend(ENV.get('BACKEND'), false /* safeMode */);
    }
  }
}

// Expects flags from URL in the format ?tfjsflags=FLAG1:1,FLAG2:true.
const TENSORFLOWJS_FLAGS_PREFIX = 'tfjsflags';
function getFeaturesFromURL(): Features {
  const features: Features = {};

  if (typeof window === 'undefined' || typeof window.location === 'undefined') {
    return features;
  }

  const urlParams = util.getQueryParams(window.location.search);
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
