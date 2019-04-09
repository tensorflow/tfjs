/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Utilities related to persistent state in the backend.
 */

/**
 * An ID to track `tf.SymbolicTensor`s and derived classes.
 * Required in different places in engine/topology.ts to identify unique
 * tensors.
 */
let _nextUniqueTensorId = 0;

export function getNextUniqueTensorId(): number {
  return _nextUniqueTensorId++;
}

const _uidPrefixes: {[prefix: string]: number} = {};

/**
 * Provides a unique UID given a string prefix.
 *
 * @param prefix
 */
export function getUid(prefix = ''): string {
  if (!(prefix in _uidPrefixes)) {
    _uidPrefixes[prefix] = 0;
  }
  _uidPrefixes[prefix] += 1;
  return prefix + _uidPrefixes[prefix].toString();
}
