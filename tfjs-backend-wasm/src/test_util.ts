import * as tf from '@tensorflow/tfjs-core';
import {resetWasmPath} from './backend_wasm';
import {setWasmPaths} from './index';

export const VALID_PREFIX = '/base/tfjs/tfjs-backend-wasm/wasm-out/';

async function cacheUrl(url: string): Promise<string> {
  const blob = await (await fetch(url)).blob();
  return URL.createObjectURL(blob);
}

let cachedUrlsPromises: Array<Promise<readonly [
  'tfjs-backend-wasm.wasm'
    | 'tfjs-backend-wasm-simd.wasm'
    | 'tfjs-backend-wasm-threaded-simd.wasm', string]>>;

if (tf.device_util.isBrowser()) {
  cachedUrlsPromises = ([
    'tfjs-backend-wasm.wasm',
    'tfjs-backend-wasm-simd.wasm',
    'tfjs-backend-wasm-threaded-simd.wasm',
  ] as const).map(async name => {
    return [name, await cacheUrl(VALID_PREFIX + name)] as const;
  });
}

/**
 * Set the wasm paths to the blob URLs that were loaded above. This is useful
 * for Karma tests because it prevents Karma from loading the same WASM files
 * every time the backend is reset (each `describe` block). This saves ~200MB
 * of network and should reduce flakiness due to network slowness / instability.
 */
export async function setupCachedWasmPaths() {
  resetWasmPath();
  if (tf.device_util.isBrowser()) {
    const cachedUrlsList = await Promise.all(cachedUrlsPromises);
    const cachedUrls = Object.fromEntries(cachedUrlsList);
    setWasmPaths(cachedUrls);
  }
}
