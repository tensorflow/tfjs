/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {dispose, Scalar} from '@tensorflow/tfjs-core';

/**
 * Logs in which values can be either numbers or Tensors (Scalars).
 *
 * Used internally.
 */
export type UnresolvedLogs = {
  [key: string]: number|Scalar;
};

/**
 * Turn any Scalar values in a Logs object into actual number values.
 *
 * @param logs The `Logs` object to be resolved in place.
 */
export async function resolveScalarsInLogs(logs: UnresolvedLogs) {
  if (logs == null) {
    return;
  }
  const promises: Array<Promise<Float32Array|Int32Array|Uint8Array>> = [];
  const keys: string[] = [];
  const scalarsToDispose: Scalar[] = [];
  for (const key in logs) {
    const value = logs[key];
    if (typeof value !== 'number') {
      const valueScalar = value;
      promises.push(valueScalar.data());
      keys.push(key);
      scalarsToDispose.push(valueScalar);
    }
  }
  if (promises.length > 0) {
    const values = await Promise.all(promises);
    for (let i = 0; i < values.length; ++i) {
      logs[keys[i]] = values[i][0];
    }
    // Dispose the original scalar tensors.
    dispose(scalarsToDispose);
  }
}

/**
 * Dispose all Tensors in an UnresolvedLogs object.
 *
 * @param logs An `UnresolvedLogs` object potentially containing `tf.Tensor`s in
 *   places where the values can be `tf.Tensor` or `number`.
 */
export function disposeTensorsInLogs(logs: UnresolvedLogs) {
  if (logs == null) {
    return;
  }
  for (const key in logs) {
    const value = logs[key];
    if (typeof value !== 'number') {
      value.dispose();
    }
  }
}

/**
 * Logs in which values can only be numbers.
 *
 * Used when calling client-provided custom callbacks.
 */
export type Logs = {
  [key: string]: number;
};
