/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* Original source: utils/generic_utils.py */

import {DataType, serialization, util} from '@tensorflow/tfjs-core';

import {AssertionError, ValueError} from '../errors';
// tslint:enable

/**
 * If `value` is an Array, equivalent to Python's `value * numValues`.
 * If `value` is not an Array, equivalent to Python's `[value] * numValues`
 */
// tslint:disable-next-line:no-any
export function pyListRepeat(value: any, numValues: number): any[] {
  if (Array.isArray(value)) {
    // tslint:disable-next-line:no-any
    let newArray: any[] = [];
    for (let i = 0; i < numValues; i++) {
      newArray = newArray.concat(value);
    }
    return newArray;
  } else {
    const newArray = new Array(numValues);
    newArray.fill(value);
    return newArray;
  }
}

export function assert(val: boolean, message?: string): void {
  if (!val) {
    throw new AssertionError(message);
  }
}

/**
 * Count the number of elements of the `array` that are equal to `reference`.
 */
export function count<T>(array: T[], refernce: T) {
  let counter = 0;
  for (const item of array) {
    if (item === refernce) {
      counter++;
    }
  }
  return counter;
}

/**
 * If an array is of length 1, just return the first element. Otherwise, return
 * the full array.
 * @param tensors
 */
export function singletonOrArray<T>(xs: T[]): T|T[] {
  if (xs.length === 1) {
    return xs[0];
  }
  return xs;
}

/**
 * Normalizes a list/tensor into a list.
 *
 * If a tensor is passed, we return
 * a list of size 1 containing the tensor.
 *
 * @param x target object to be normalized.
 */
// tslint:disable-next-line:no-any
export function toList(x: any): any[] {
  if (Array.isArray(x)) {
    return x;
  }
  return [x];
}

/**
 * Generate a UID for a list
 */
// tslint:disable-next-line:no-any
export function objectListUid(objs: any|any[]): string {
  const objectList = toList(objs);
  let retVal = '';
  for (const obj of objectList) {
    if (obj.id == null) {
      throw new ValueError(
          `Object ${obj} passed to objectListUid without an id`);
    }
    if (retVal !== '') {
      retVal = retVal + ', ';
    }
    retVal = retVal + Math.abs(obj.id);
  }
  return retVal;
}
/**
 * Converts string to snake-case.
 * @param name
 */
export function toSnakeCase(name: string): string {
  const intermediate = name.replace(/(.)([A-Z][a-z0-9]+)/g, '$1_$2');
  const insecure =
      intermediate.replace(/([a-z])([A-Z])/g, '$1_$2').toLowerCase();
  /*
   If the class is private the name starts with "_" which is not secure
   for creating scopes. We prefix the name with "private" in this case.
   */
  if (insecure[0] !== '_') {
    return insecure;
  }
  return 'private' + insecure;
}

export function toCamelCase(identifier: string): string {
  // quick return for empty string or single character strings
  if (identifier.length <= 1) {
    return identifier;
  }
  // Check for the underscore indicating snake_case
  if (identifier.indexOf('_') === -1) {
    return identifier;
  }
  return identifier.replace(/[_]+(\w|$)/g, (m, p1) => p1.toUpperCase());
}

// tslint:disable-next-line:no-any
let _GLOBAL_CUSTOM_OBJECTS = {} as {[objName: string]: any};

export function serializeKerasObject(instance: serialization.Serializable):
    serialization.ConfigDictValue {
  if (instance === null || instance === undefined) {
    return null;
  }
  const dict: serialization.ConfigDictValue = {};
  dict['className'] = instance.getClassName();
  dict['config'] = instance.getConfig();
  return dict;
}

/**
 * Replace ndarray-style scalar objects in serialization objects with numbers.
 *
 * Background: In some versions of tf.keras, certain scalar values in the HDF5
 * model save file can be serialized as: `{'type': 'ndarray', 'value': num}`,
 * where in `num` is a plain number. This method converts such serialization
 * to a `number`.
 *
 * @param config The keras-format serialization object to be processed
 *   (in place).
 */
function convertNDArrayScalarsInConfig(config: serialization.ConfigDictValue):
    void {
  if (config == null || typeof config !== 'object') {
    return;
  } else if (Array.isArray(config)) {
    config.forEach(configItem => convertNDArrayScalarsInConfig(configItem));
  } else {
    const fields = Object.keys(config);
    for (const field of fields) {
      const value = config[field];
      if (value != null && typeof value === 'object') {
        if (!Array.isArray(value) && value['type'] === 'ndarray' &&
            typeof value['value'] === 'number') {
          config[field] = value['value'];
        } else {
          convertNDArrayScalarsInConfig(value as serialization.ConfigDict);
        }
      }
    }
  }
}

/**
 * Deserialize a saved Keras Object
 * @param identifier either a string ID or a saved Keras dictionary
 * @param moduleObjects a list of Python class names to object constructors
 * @param customObjects a list of Python class names to object constructors
 * @param printableModuleName debug text for the object being reconstituted
 * @param fastWeightInit Optional flag to use fast weight initialization
 *   during deserialization. This is applicable to cases in which
 *   the initialization will be immediately overwritten by loaded weight
 *   values. Default: `false`.
 * @returns a TensorFlow.js Layers object
 */
// tslint:disable:no-any
export function deserializeKerasObject(
    identifier: string|serialization.ConfigDict,
    moduleObjects = {} as {[objName: string]: any},
    customObjects = {} as {[objName: string]: any},
    printableModuleName = 'object', fastWeightInit = false): any {
  // tslint:enable
  if (typeof identifier === 'string') {
    const functionName = identifier;
    let fn;
    if (functionName in customObjects) {
      fn = customObjects[functionName];
    } else if (functionName in _GLOBAL_CUSTOM_OBJECTS) {
      fn = _GLOBAL_CUSTOM_OBJECTS[functionName];
    } else {
      fn = moduleObjects[functionName];
      if (fn == null) {
        throw new ValueError(
            `Unknown ${printableModuleName}: ${identifier}. ` +
            `This may be due to one of the following reasons:\n` +
            `1. The ${printableModuleName} is defined in Python, in which ` +
            `case it needs to be ported to TensorFlow.js or your JavaScript ` +
            `code.\n` +
            `2. The custom ${printableModuleName} is defined in JavaScript, ` +
            `but is not registered properly with ` +
            `tf.serialization.registerClass().`);
        // TODO(cais): Add link to tutorial page on custom layers.
      }
    }
    return fn;
  } else {
    // In this case we are dealing with a Keras config dictionary.
    const config = identifier;
    if (config['className'] == null || config['config'] == null) {
      throw new ValueError(
          `${printableModuleName}: Improper config format: ` +
          `${JSON.stringify(config)}.\n` +
          `'className' and 'config' must set.`);
    }
    const className = config['className'] as string;
    let cls, fromConfig;
    if (className in customObjects) {
      [cls, fromConfig] = customObjects[className];
    } else if (className in _GLOBAL_CUSTOM_OBJECTS) {
      [cls, fromConfig] = _GLOBAL_CUSTOM_OBJECTS['className'];
    } else if (className in moduleObjects) {
      [cls, fromConfig] = moduleObjects[className];
    }
    if (cls == null) {
      throw new ValueError(
          `Unknown ${printableModuleName}: ${className}. ` +
          `This may be due to one of the following reasons:\n` +
          `1. The ${printableModuleName} is defined in Python, in which ` +
          `case it needs to be ported to TensorFlow.js or your JavaScript ` +
          `code.\n` +
          `2. The custom ${printableModuleName} is defined in JavaScript, ` +
          `but is not registered properly with ` +
          `tf.serialization.registerClass().`);
      // TODO(cais): Add link to tutorial page on custom layers.
    }
    if (fromConfig != null) {
      // Porting notes: Instead of checking to see whether fromConfig accepts
      // customObjects, we create a customObjects dictionary and tack it on to
      // config['config'] as config['config'].customObjects. Objects can use it,
      // if they want.

      // tslint:disable-next-line:no-any
      const customObjectsCombined = {} as {[objName: string]: any};
      for (const key of Object.keys(_GLOBAL_CUSTOM_OBJECTS)) {
        customObjectsCombined[key] = _GLOBAL_CUSTOM_OBJECTS[key];
      }
      for (const key of Object.keys(customObjects)) {
        customObjectsCombined[key] = customObjects[key];
      }
      // Add the customObjects to config
      const nestedConfig = config['config'] as serialization.ConfigDict;
      nestedConfig['customObjects'] = customObjectsCombined;

      const backupCustomObjects = {..._GLOBAL_CUSTOM_OBJECTS};
      for (const key of Object.keys(customObjects)) {
        _GLOBAL_CUSTOM_OBJECTS[key] = customObjects[key];
      }
      convertNDArrayScalarsInConfig(config['config']);
      const returnObj =
          fromConfig(cls, config['config'], customObjects, fastWeightInit);
      _GLOBAL_CUSTOM_OBJECTS = {...backupCustomObjects};

      return returnObj;
    } else {
      // Then `cls` may be a function returning a class.
      // In this case by convention `config` holds
      // the kwargs of the function.
      const backupCustomObjects = {..._GLOBAL_CUSTOM_OBJECTS};
      for (const key of Object.keys(customObjects)) {
        _GLOBAL_CUSTOM_OBJECTS[key] = customObjects[key];
      }
      // In python this is **config['config'], for tfjs-layers we require
      // classes that use this fall-through construction method to take
      // a config interface that mimics the expansion of named parameters.
      const returnObj = new cls(config['config']);
      _GLOBAL_CUSTOM_OBJECTS = {...backupCustomObjects};
      return returnObj;
    }
  }
}

/**
 * Compares two numbers for sorting.
 * @param a
 * @param b
 */
export function numberCompare(a: number, b: number) {
  return (a < b) ? -1 : ((a > b) ? 1 : 0);
}

/**
 * Comparison of two numbers for reverse sorting.
 * @param a
 * @param b
 */
export function reverseNumberCompare(a: number, b: number) {
  return -1 * numberCompare(a, b);
}

/**
 * Convert a string into the corresponding DType.
 * @param dtype
 * @returns An instance of DType.
 */
export function stringToDType(dtype: string): DataType {
  switch (dtype) {
    case 'float32':
      return 'float32';
    default:
      throw new ValueError(`Invalid dtype: ${dtype}`);
  }
}

/**
 * Test the element-by-element equality of two Arrays of strings.
 * @param xs First array of strings.
 * @param ys Second array of strings.
 * @returns Wether the two arrays are all equal, element by element.
 */
export function stringsEqual(xs: string[], ys: string[]): boolean {
  if (xs == null || ys == null) {
    return xs === ys;
  }
  if (xs.length !== ys.length) {
    return false;
  }
  for (let i = 0; i < xs.length; ++i) {
    if (xs[i] !== ys[i]) {
      return false;
    }
  }
  return true;
}

/**
 * Get the unique elements of an array.
 * @param xs Array.
 * @returns An Array consisting of the unique elements in `xs`.
 */
export function unique<T>(xs: T[]): T[] {
  if (xs == null) {
    return xs;
  }
  const out: T[] = [];
  // TODO(cais): Maybe improve performance by sorting.
  for (const x of xs) {
    if (out.indexOf(x) === -1) {
      out.push(x);
    }
  }
  return out;
}

/**
 * Determine if an Object is empty (i.e., does not have own properties).
 * @param obj Object
 * @returns Whether the Object is empty.
 * @throws ValueError: If object is `null` or `undefined`.
 */
export function isObjectEmpty(obj: {}): boolean {
  if (obj == null) {
    throw new ValueError(`Invalid value in obj: ${JSON.stringify(obj)}`);
  }
  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      return false;
    }
  }
  return true;
}

/**
 * Helper function used to build type union/enum run-time checkers.
 * @param values The list of allowed values.
 * @param label A string name for the type
 * @param value The value to test.
 * @throws ValueError: If the value is not in values nor `undefined`/`null`.
 */
export function checkStringTypeUnionValue(
    values: string[], label: string, value: string): void {
  if (value == null) {
    return;
  }
  if (values.indexOf(value) < 0) {
    throw new ValueError(`${value} is not a valid ${label}.  Valid values are ${
        values} or null/undefined.`);
  }
}

/**
 * Helper function for verifying the types of inputs.
 *
 * Ensures that the elements of `x` are all of type `expectedType`.
 * Also verifies that the length of `x` is within bounds.
 *
 * @param x Object to test.
 * @param expectedType The string expected type of all of the elements in the
 * Array.
 * @param minLength Return false if x.length is less than this.
 * @param maxLength Return false if x.length is greater than this.
 * @returns true if and only if `x` is an `Array<expectedType>` with
 * length >= `minLength` and <= `maxLength`.
 */
// tslint:disable:no-any
export function checkArrayTypeAndLength(
    x: any, expectedType: string, minLength = 0,
    maxLength = Infinity): boolean {
  assert(minLength >= 0);
  assert(maxLength >= minLength);
  return (
      Array.isArray(x) && x.length >= minLength && x.length <= maxLength &&
      x.every(e => typeof e === expectedType));
}
// tslint:enable:no-any

/**
 * Assert that a value or an array of value are positive integer.
 *
 * @param value The value being asserted on. May be a single number or an array
 *   of numbers.
 * @param name Name of the value, used to make the error message.
 */
export function assertPositiveInteger(value: number|number[], name: string) {
  if (Array.isArray(value)) {
    util.assert(
        value.length > 0, () => `${name} is unexpectedly an empty array.`);
    value.forEach(
        (v, i) => assertPositiveInteger(v, `element ${i + 1} of ${name}`));
  } else {
    util.assert(
        Number.isInteger(value) && value > 0,
        () => `Expected ${name} to be a positive integer, but got ` +
            `${formatAsFriendlyString(value)}.`);
  }
}

/**
 * Format a value into a display-friendly, human-readable fashion.
 *
 * - `null` is formatted as `'null'`
 * - Strings are formated with flanking pair of quotes.
 * - Arrays are formatted with flanking pair of square brackets.
 *
 * @param value The value to display.
 * @return Formatted string.
 */
// tslint:disable-next-line:no-any
export function formatAsFriendlyString(value: any): string {
  if (value === null) {
    return 'null';
  } else if (Array.isArray(value)) {
    return '[' + value.map(v => formatAsFriendlyString(v)).join(',') + ']';
  } else if (typeof value === 'string') {
    return `"${value}"`;
  } else {
    return `${value}`;
  }
}
