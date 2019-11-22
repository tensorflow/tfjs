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
import {env} from '../environment';

import {Platform} from './platform';

// We are wrapping this within an object so it can be stubbed by Jasmine.
export const getNodeFetch = {
  // tslint:disable-next-line:no-require-imports
  importFetch: () => require('node-fetch')
};

type FetchFn = (url: string, init?: RequestInit) => Promise<Response>;
let systemFetch: FetchFn;
// These getters and setters are for testing so we don't export a mutable
// variable.
export function resetSystemFetch() {
  systemFetch = null;
}
export function setSystemFetch(fetchFn: FetchFn) {
  systemFetch = fetchFn;
}
export function getSystemFetch(): FetchFn {
  return systemFetch;
}

export class PlatformNode implements Platform {
  private textEncoder: TextEncoder;
  // tslint:disable-next-line:no-any
  util: any;

  constructor() {
    // tslint:disable-next-line:no-require-imports
    this.util = require('util');
    // According to the spec, the built-in encoder can do only UTF-8 encoding.
    // https://developer.mozilla.org/en-US/docs/Web/API/TextEncoder/TextEncoder
    this.textEncoder = new this.util.TextEncoder();
  }

  fetch(path: string, requestInits?: RequestInit): Promise<Response> {
    if (env().global.fetch != null) {
      return env().global.fetch(path, requestInits);
    }

    if (systemFetch == null) {
      systemFetch = getNodeFetch.importFetch();
    }
    return systemFetch(path, requestInits);
  }

  now(): number {
    const time = process.hrtime();
    return time[0] * 1000 + time[1] / 1000000;
  }

  encode(text: string, encoding: string): Uint8Array {
    if (encoding !== 'utf-8' && encoding !== 'utf8') {
      throw new Error(
          `Node built-in encoder only supports utf-8, but got ${encoding}`);
    }
    return this.textEncoder.encode(text);
  }
  decode(bytes: Uint8Array, encoding: string): string {
    if (bytes.length === 0) {
      return '';
    }
    return new this.util.TextDecoder(encoding).decode(bytes);
  }
}

if (env().get('IS_NODE')) {
  env().setPlatform('node', new PlatformNode());
}
