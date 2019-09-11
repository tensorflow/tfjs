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
import {ENV} from '../../environment';

import {cleanupDOMCanvasWebGLRenderingContext, createDOMCanvasWebGLRenderingContext} from './canvas_util';
import {callAndCheck, checkWebGLError} from './webgl_util';

let count = 0;
const contexts: {[key: string]: WebGLRenderingContext} = {};
let contextFactory: (version: number) => WebGLRenderingContext = null;
let contextCleanup: (context: WebGLRenderingContext) => void = null;

/**
 * Sets the callback for creating new WebGLRenderingContext instances.
 * @param factory The callback function that returns a context instance.
 */
export function setContextFactory(
    factory: (version: number) => WebGLRenderingContext) {
  contextFactory = factory;
}

/**
 * Sets the callback for cleaning up WebGLRenderingContext instances.
 * @param cleanup The callback function to cleanup the passed in context
 *     instance.
 */
export function setContextCleanup(
    cleanup: (context: WebGLRenderingContext) => void) {
  contextCleanup = cleanup;
}

/**
 * Returns the current WebGLRenderingContext based on the ENV flag for
 * 'WEBGL_VERSION'.
 */
export function getActiveContext(): WebGLRenderingContext {
  return getContextByVersion(ENV.getNumber('WEBGL_VERSION'));
}

/**
 * Returns the WebGLRenderingContext for a given version number.
 * @param version The specific version of WebGL to request.
 */
export function getContextByVersion(version: number): WebGLRenderingContext {
  // Default to browser context creation is running in the browser.
  if (contextFactory == null) {
    if (ENV.getBool('IS_BROWSER')) {
      // TODO(kreeger): Is there a better place to register this?
      contextFactory = createDOMCanvasWebGLRenderingContext;
    } else {
      throw new Error('Default WebGLRenderingContext factory was not set!');
    }
  }

  if (!(version in contexts)) {
    contexts[version] = traceGLCalls(contextFactory(version), ++count);
    bootstrapWebGLContext(contexts[version]);
    checkWebGLError(contexts[version]);
  }

  const gl = contexts[version];
  if (gl.isContextLost()) {
    checkWebGLError(contexts[version]);
    disposeWebGLContext(version);
    return getContextByVersion(version);
  }
  checkWebGLError(contexts[version]);
  return contexts[version];
}

function disposeWebGLContext(version: number) {
  if ((version in contexts)) {
    if (contextCleanup == null) {
      if (ENV.getBool('IS_BROWSER')) {
        // TODO(kreeger): Is there a better place to register this?
        contextCleanup = cleanupDOMCanvasWebGLRenderingContext;
      }
    }
    if (contextCleanup != null) {
      contextCleanup(contexts[version]);
    }
    delete contexts[version];
  }
}

function bootstrapWebGLContext(gl: WebGLRenderingContext) {
  // TODO - check GL calls here too.
  callAndCheck(gl, ENV.getBool('DEBUG'), () => gl.disable(gl.DEPTH_TEST));
  // gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.STENCIL_TEST);
  gl.disable(gl.BLEND);
  gl.disable(gl.DITHER);
  gl.disable(gl.POLYGON_OFFSET_FILL);
  gl.disable(gl.SAMPLE_COVERAGE);
  gl.enable(gl.SCISSOR_TEST);
  gl.enable(gl.CULL_FACE);
  gl.cullFace(gl.BACK);
}

function traceGLCalls(ctx: WebGLRenderingContext, idx: number) {
  const handler = {
    // tslint:disable-next-line:no-any
    get(target: any, prop: PropertyKey, receiver: any): any {
      const propValue = target[prop];

      if (typeof (propValue) === 'function') {
        console.log(
            '    gl.' + prop.toString() + ' = ' + target.constructor.name +
            ' ' + idx);
        // tslint:disable-next-line:only-arrow-functions
        return function() {
          return propValue.apply(target, arguments);
        };
      }
      return propValue;
    },
  };
  return new Proxy(ctx, handler);
}
