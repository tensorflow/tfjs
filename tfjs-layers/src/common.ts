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
import {VALID_DATA_FORMAT_VALUES, VALID_PADDING_MODE_VALUES, VALID_POOL_MODE_VALUES} from './keras_format/common';
import {checkStringTypeUnionValue} from './utils/generic_utils';

// A map from the requested scoped name of a Tensor to the number of Tensors
// wanting that name so far.  This allows enforcing name uniqueness by appending
// an incrementing index, e.g. scope/name, scope/name_1, scope/name_2, etc.
const nameMap: Map<string, number> = new Map<string, number>();

export function checkDataFormat(value?: string): void {
  checkStringTypeUnionValue(VALID_DATA_FORMAT_VALUES, 'DataFormat', value);
}

export function checkPaddingMode(value?: string): void {
  checkStringTypeUnionValue(VALID_PADDING_MODE_VALUES, 'PaddingMode', value);
}

export function checkPoolMode(value?: string): void {
  checkStringTypeUnionValue(VALID_POOL_MODE_VALUES, 'PoolMode', value);
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
 * Get the name a Tensor (or Variable) would have if not uniqueified.
 * @param tensorName
 * @return Scoped name string.
 */
export function getScopedTensorName(tensorName: string): string {
  if (!isValidTensorName(tensorName)) {
    throw new Error('Not a valid tensor name: \'' + tensorName + '\'');
  }
  return currentNameScopePrefix() + tensorName;
}

/**
 * Get unique names for Tensors and Variables.
 * @param scopedName The fully-qualified name of the Tensor, i.e. as produced by
 *  `getScopedTensorName()`.
 * @return A unique version of the given fully scoped name.
 *   If this is the first time that the scoped name is seen in this session,
 *   then the given `scopedName` is returned unaltered.  If the same name is
 *   seen again (producing a collision), an incrementing suffix is added to the
 *   end of the name, so it takes the form 'scope/name_1', 'scope/name_2', etc.
 */
export function getUniqueTensorName(scopedName: string): string {
  if (!isValidTensorName(scopedName)) {
    throw new Error('Not a valid tensor name: \'' + scopedName + '\'');
  }
  if (!nameMap.has(scopedName)) {
    nameMap.set(scopedName, 0);
  }
  const index = nameMap.get(scopedName);
  nameMap.set(scopedName, nameMap.get(scopedName) + 1);

  if (index > 0) {
    const result = `${scopedName}_${index}`;
    // Mark the composed name as used in case someone wants
    // to call getUniqueTensorName("name_1").
    nameMap.set(result, 1);
    return result;
  } else {
    return scopedName;
  }
}

const tensorNameRegex = new RegExp(/^[A-Za-z0-9][-A-Za-z0-9\._\/]*$/);

/**
 * Determine whether a string is a valid tensor name.
 * @param name
 * @returns A Boolean indicating whether `name` is a valid tensor name.
 */
export function isValidTensorName(name: string): boolean {
  return !!name.match(tensorNameRegex);
}
