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
import {Platform} from '@tensorflow/tfjs-core';

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

export class PlatformReactNative implements Platform {
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
   */
  async fetch(path: string, init?: RequestInit): Promise<Response> {
    return new Promise((resolve, reject) => {
      const request = new Request(path, init);
      const xhr = new XMLHttpRequest();

      xhr.onload = () => {
        const options = {
          status: xhr.status,
          statusText: xhr.statusText,
          headers: parseHeaders(xhr.getAllResponseHeaders() || ''),
          url: '',
        };
        options.url = 'responseURL' in xhr ?
            xhr.responseURL :
            options.headers.get('X-Request-URL');
        //@ts-ignore — ts belives the latter case will never occur.
        const body = 'response' in xhr ? xhr.response : xhr.responseText;
        resolve(new Response(body, options));
      };

      xhr.onerror = () => reject(new TypeError('Network request failed'));
      xhr.ontimeout = () => reject(new TypeError('Network request failed'));

      xhr.open(request.method, request.url, true);

      if (request.credentials === 'include') {
        xhr.withCredentials = true;
      } else if (request.credentials === 'omit') {
        xhr.withCredentials = false;
      }

      // IO handlers are responsible for explicitly setting this header
      // to 'arraybuffer' when loading binary files.
      if (request.headers.get('responseType')) {
        xhr.responseType =
            request.headers.get('responseType') as XMLHttpRequestResponseType;
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

  /**
   * Encode the provided string into an array of bytes using the provided
   * encoding.
   */
  encode(text: string, encoding: string): Uint8Array {
    throw new Error('not yet implemented');
  }
  /** Decode the provided bytes into a string using the provided encoding. */
  decode(bytes: Uint8Array, encoding: string): string {
    throw new Error('not yet implemented');
  }
}

tf.ENV.registerFlag(
    'IS_REACT_NATIVE', () => navigator && navigator.product === 'ReactNative');

if (tf.ENV.getBool('IS_REACT_NATIVE')) {
  tf.setPlatform('react-native', new PlatformReactNative());
}
