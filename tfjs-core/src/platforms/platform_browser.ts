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

import '../flags';

import {env} from '../environment';
import {BrowserIndexedDB, BrowserIndexedDBManager} from '../io/indexed_db';
import {BrowserLocalStorage, BrowserLocalStorageManager} from '../io/local_storage';
import {ModelStoreManagerRegistry} from '../io/model_management';

import {Platform} from './platform';
import {isTypedArrayBrowser} from './is_typed_array_browser';

export class PlatformBrowser implements Platform {
  // According to the spec, the built-in encoder can do only UTF-8 encoding.
  // https://developer.mozilla.org/en-US/docs/Web/API/TextEncoder/TextEncoder
  private textEncoder: TextEncoder;

  // For setTimeoutCustom
  private readonly messageName = 'setTimeoutCustom';
  private functionRefs: Function[] = [];
  private handledMessageCount = 0;
  private hasEventListener = false;

  fetch(path: string, init?: RequestInit): Promise<Response> {
    return fetch(path, init);
  }

  now(): number {
    return performance.now();
  }

  encode(text: string, encoding: string): Uint8Array {
    if (encoding !== 'utf-8' && encoding !== 'utf8') {
      throw new Error(
          `Browser's encoder only supports utf-8, but got ${encoding}`);
    }
    if (this.textEncoder == null) {
      this.textEncoder = new TextEncoder();
    }
    return this.textEncoder.encode(text);
  }
  decode(bytes: Uint8Array, encoding: string): string {
    return new TextDecoder(encoding).decode(bytes);
  }

  // If the setTimeout nesting level is greater than 5 and timeout is less
  // than 4ms, timeout will be clamped to 4ms, which hurts the perf.
  // Interleaving window.postMessage and setTimeout will trick the browser and
  // avoid the clamp.
  setTimeoutCustom(functionRef: Function, delay: number): void {
    if (typeof window === 'undefined' ||
        !env().getBool('USE_SETTIMEOUTCUSTOM')) {
      setTimeout(functionRef, delay);
      return;
    }

    this.functionRefs.push(functionRef);
    setTimeout(() => {
      window.postMessage(
          {name: this.messageName, index: this.functionRefs.length - 1}, '*');
    }, delay);

    if (!this.hasEventListener) {
      this.hasEventListener = true;
      window.addEventListener('message', (event: MessageEvent) => {
        if (event.source === window && event.data.name === this.messageName) {
          event.stopPropagation();
          const functionRef = this.functionRefs[event.data.index];
          functionRef();
          this.handledMessageCount++;
          if (this.handledMessageCount === this.functionRefs.length) {
            this.functionRefs = [];
            this.handledMessageCount = 0;
          }
        }
      }, true);
    }
  }

  isTypedArray(a: unknown): a is Uint8Array | Float32Array | Int32Array
    | Uint8ClampedArray {
    return isTypedArrayBrowser(a);
  }
}

if (env().get('IS_BROWSER')) {
  env().setPlatform('browser', new PlatformBrowser());

  // Register LocalStorage IOHandler
  try {
    ModelStoreManagerRegistry.registerManager(
        BrowserLocalStorage.URL_SCHEME, new BrowserLocalStorageManager());
  } catch (err) {
  }

  // Register IndexedDB IOHandler
  try {
    ModelStoreManagerRegistry.registerManager(
        BrowserIndexedDB.URL_SCHEME, new BrowserIndexedDBManager());
  } catch (err) {
  }
}
