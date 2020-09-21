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

import '@tensorflow/tfjs-backend-cpu';
import {GPGPUContext, MathBackendWebGL, setWebGLContext} from '@tensorflow/tfjs-backend-webgl';
import * as tf from '@tensorflow/tfjs-core';
import {Platform} from '@tensorflow/tfjs-core';
import {Buffer} from 'buffer';
import {GLView} from 'expo-gl';
import {Platform as RNPlatform} from 'react-native';

let debugMode_ = false;
export function setDebugMode(debugMode: boolean) {
  debugMode_ = debugMode_;
}

export function getDebugMode() {
  return debugMode_;
}

// See implemetation note on fetch
// tslint:disable-next-line:max-line-length
// https://github.com/facebook/react-native/blob/0ee5f68929610106ee6864baa04ea90be0fc5160/Libraries/vendor/core/whatwg-fetch.js#L421
function parseHeaders(rawHeaders: string) {
  const headers = new Headers();
  // Replace instances of \r\n and \n followed by at least one space or
  // horizontal tab with a space https://tools.ietf.org/html/rfc7230#section-3.2
  const preProcessedHeaders = rawHeaders.replace(/\r?\n[\t ]+/g, ' ');
  preProcessedHeaders.split(/\r?\n/).forEach(line => {
    const parts = line.split(':');
    const key = parts.shift().trim();
    if (key) {
      const value = parts.join(':').trim();
      headers.append(key, value);
    }
  });
  return headers;
}

// Implementation note: This is a patch of react-native's fetch implementation
// tslint:disable-next-line:max-line-length
// https://github.com/facebook/react-native/blob/0ee5f68929610106ee6864baa04ea90be0fc5160/Libraries/vendor/core/whatwg-fetch.js#L484
//
// The response object supplied by fetch does not implement arrayBuffer()
// FileReader.readAsArrayBuffer is not implemented.
// tslint:disable-next-line:max-line-length
// https://github.com/facebook/react-native/blob/d7a5e3e215eedb7377a86f172e0619403e20c2b8/Libraries/Blob/FileReader.js#L83
//
// However if one uses XMLHttpRequest directly and set the responseType
// correctly before making the request. The returned response object will have
// a working arrayBuffer method that can be used downstraeam.

/**
 * Makes an HTTP request.
 * @param path The URL path to make a request to
 * @param init The request init. See init here:
 *     https://developer.mozilla.org/en-US/docs/Web/API/Request/Request
 * @param options A RequestDetails object.
 *    - __options.isBinary__ boolean indicating whether this request is for a
 *     binary file.
 *
 * @doc {heading: 'Platform helpers', subheading: 'http'}
 */
export async function fetch(
    path: string, init?: RequestInit,
    options?: tf.io.RequestDetails): Promise<Response> {
  return new Promise((resolve, reject) => {
    const request = new Request(path, init);
    const xhr = new XMLHttpRequest();

    xhr.onload = () => {
      const reqOptions = {
        status: xhr.status,
        statusText: xhr.statusText,
        headers: parseHeaders(xhr.getAllResponseHeaders() || ''),
        url: '',
      };
      reqOptions.url = 'responseURL' in xhr ?
          xhr.responseURL :
          reqOptions.headers.get('X-Request-URL');

      //@ts-ignore — ts belives the latter case will never occur.
      const body = 'response' in xhr ? xhr.response : xhr.responseText;

      resolve(new Response(body, reqOptions));
    };

    xhr.onerror = () => reject(new TypeError('Network request failed'));
    xhr.ontimeout = () => reject(new TypeError('Network request failed'));

    xhr.open(request.method, request.url, true);

    if (request.credentials === 'include') {
      xhr.withCredentials = true;
    } else if (request.credentials === 'omit') {
      xhr.withCredentials = false;
    }

    if (options != null && options.isBinary) {
      // In react native We need to set the response type to arraybuffer when
      // fetching binary resources in order for `.arrayBuffer` to work correctly
      // on the response.
      xhr.responseType = 'arraybuffer';
    }

    request.headers.forEach((value: string, name: string) => {
      xhr.setRequestHeader(name, value);
    });

    xhr.send(
        //@ts-ignore
        typeof request._bodyInit === 'undefined' ? null : request._bodyInit,
    );
  });
}

export class PlatformReactNative implements Platform {
  /**
   * Makes an HTTP request.
   *
   * see @fetch docs above.
   */
  async fetch(
      path: string, init?: RequestInit, options?: tf.io.RequestDetails) {
    return fetch(path, init, options);
  }

  /**
   * Encode the provided string into an array of bytes using the provided
   * encoding.
   */
  encode(text: string, encoding: string): Uint8Array {
    // See https://www.w3.org/TR/encoding/#utf-16le
    if (encoding === 'utf-16') {
      encoding = 'utf16le';
    }
    return new Uint8Array(Buffer.from(text, encoding as BufferEncoding));
  }
  /** Decode the provided bytes into a string using the provided encoding. */
  decode(bytes: Uint8Array, encoding: string): string {
    // See https://www.w3.org/TR/encoding/#utf-16le
    if (encoding === 'utf-16') {
      encoding = 'utf16le';
    }
    return Buffer.from(bytes).toString(encoding);
  }

  now(): number {
    //@ts-ignore
    if (global.nativePerformanceNow) {
      //@ts-ignore
      return global.nativePerformanceNow();
    }
    return Date.now();
  }
}

function setupGlobals() {
  global.Buffer = Buffer;
}

function registerWebGLBackend() {
  try {
    const PRIORITY = 5;
    tf.registerBackend('rn-webgl', async () => {
      const glContext = await GLView.createContextAsync();

      // ExpoGl getBufferSubData is not implemented yet (throws an exception).
      tf.env().set('WEBGL_BUFFER_SUPPORTED', false);

      //
      // Mock extension support for EXT_color_buffer_float and
      // EXT_color_buffer_half_float on the expo-gl context object.
      // In react native we do not have to get a handle to the extension
      // in order to use the functionality of that extension on the device.
      //
      // This code block makes iOS and Android devices pass the extension checks
      // used in core. After those are done core will actually test whether
      // we can render/download float or half float textures.
      //
      // We can remove this block once we upstream checking for these
      // extensions in expo.
      //
      // TODO look into adding support for checking these extensions in expo-gl
      //
      //@ts-ignore
      const getExt = glContext.getExtension.bind(glContext);
      const shimGetExt = (name: string) => {
        if (name === 'EXT_color_buffer_float') {
          if (RNPlatform.OS === 'ios') {
            // iOS does not support EXT_color_buffer_float
            return null;
          } else {
            return {};
          }
        }

        if (name === 'EXT_color_buffer_half_float') {
          return {};
        }
        return getExt(name);
      };

      //
      // Manually make 'read' synchronous. glContext has a defined gl.fenceSync
      // function that throws a "Not implemented yet" exception so core
      // cannot properly detect that it is not supported. We mock
      // implementations of gl.fenceSync and gl.clientWaitSync
      // TODO remove once fenceSync and clientWaitSync is implemented upstream.
      //
      const shimFenceSync = () => {
        return {};
      };
      const shimClientWaitSync = () => glContext.CONDITION_SATISFIED;

      // @ts-ignore
      glContext.getExtension = shimGetExt.bind(glContext);
      glContext.fenceSync = shimFenceSync.bind(glContext);
      glContext.clientWaitSync = shimClientWaitSync.bind(glContext);

      // Set the WebGLContext before flag evaluation
      setWebGLContext(2, glContext);
      const context = new GPGPUContext();
      const backend = new MathBackendWebGL(context);

      return backend;
    }, PRIORITY);

    // Register all the webgl kernels on the rn-webgl backend
    // TODO: Use tf.copyRegisteredKernels once synced to tfjs-core 2.5.0.
    // tf.copyRegisteredKernels('webgl', 'rn-webgl');
    const kernels = tf.getKernelsForBackend('webgl');
    kernels.forEach(kernelConfig => {
      const newKernelConfig =
          Object.assign({}, kernelConfig, {backendName: 'rn-webgl'});
      tf.registerKernel(newKernelConfig);
    });
  } catch (e) {
    throw (new Error(`Failed to register Webgl backend: ${e.message}`));
  }
}

tf.env().registerFlag(
    'IS_REACT_NATIVE', () => navigator && navigator.product === 'ReactNative');

if (tf.env().getBool('IS_REACT_NATIVE')) {
  setupGlobals();
  registerWebGLBackend();
  tf.setPlatform('react-native', new PlatformReactNative());
}
