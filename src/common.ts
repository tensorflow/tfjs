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
 * Common functions for TensorFlow.js Layers.
 */
import {SerializableEnumRegistry} from './utils/generic_utils';

const nameMap: Map<string, number> = new Map<string, number>();

// TODO(cais): Perhaps move the enums to a more suitable place, e.g.,
//   constants.ts.
export enum DataFormat {
  CHANNEL_FIRST,
  CHANNEL_LAST
}
SerializableEnumRegistry.register('data_format', {
  'channels_first': DataFormat.CHANNEL_FIRST,
  'channels_last': DataFormat.CHANNEL_LAST
});

export enum PaddingMode {
  VALID,
  SAME,
  CASUAL,
}
SerializableEnumRegistry.register(
    'padding', {'valid': PaddingMode.VALID, 'same': PaddingMode.SAME});

export enum PoolMode {
  MAX,
  AVG,
}

const _nameScopeStack: string[] = [];
const _nameScopeDivider = '/';

/**
 * Enter namescope, which can be nested.
 */
export function nameScope<T>(name: string, fn: () => T): T {
  _nameScopeStack.push(name);
  try {
    const val: T = fn();
    _nameScopeStack.pop();
    return val;
  } catch (e) {
    _nameScopeStack.pop();
    throw e;
  }
}

/**
 * Get the current namescope as a flat, concatenated string.
 */
function currentNameScopePrefix(): string {
  if (_nameScopeStack.length === 0) {
    return '';
  } else {
    return _nameScopeStack.join(_nameScopeDivider) + _nameScopeDivider;
  }
}

/**
 * Get unique names for Tensors (and Variables).
 * @param prefix
 * @return Unique name string.
 */
export function getUniqueTensorName(prefix: string): string {
  if (!isValidTensorName(prefix)) {
    throw new Error('Not a valid tensor name: \'' + prefix + '\'');
  }

  prefix = currentNameScopePrefix() + prefix;

  if (!nameMap.has(prefix)) {
    nameMap.set(prefix, 0);
  }
  const index = nameMap.get(prefix);
  nameMap.set(prefix, nameMap.get(prefix) + 1);

  if (index > 0) {
    return prefix + '_' + index;
  } else {
    return prefix;
  }
}

const tensorNameRegex = new RegExp(/^[A-Za-z][A-Za-z0-9\._\/]*$/);

/**
 * Determine whether a string is a valid tensor name.
 * @param name
 * @returns A Boolean indicating whether `name` is a valid tensor name.
 */
export function isValidTensorName(name: string): boolean {
  return name.match(tensorNameRegex) ? true : false;
}
