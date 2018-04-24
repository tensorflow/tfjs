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

// tslint:disable:max-line-length
import {Tensor} from '@tensorflow/tfjs-core';

import {AssertionError, AttributeError, IndexError, ValueError} from '../errors';
import {ConfigDict, ConfigDictValue, Constructor, DType, FromConfigMethod, Serializable, Shape} from '../types';



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

/**
 * Equivalent to Python's getattr() built-in function.
 * @param obj
 * @param attrName The name of the attribute to retrieve.
 * @param defaultValue Default value to use if attrName doesn't exist in the
 *   object.
 */
// tslint:disable-next-line:no-any
export function pyGetAttr<T>(obj: any, attrName: string, defaultValue?: T): T {
  if (attrName in obj) {
    return obj[attrName];
  }
  if (defaultValue === undefined) {
    throw new AttributeError(
        'pyGetAttr: Attempting to get attribute ' + attrName +
        'with no default value defined');
  }
  return defaultValue;
}

/**
 * Python allows indexing into a list from the end using negative values. This
 * utility functions translates an index into a list into a non-negative index,
 * allowing for negative indices, just like Python.
 *
 * @param x An array.
 * @param index The index to normalize.
 * @return A non-negative index, within range.
 * @exception IndexError if index is not within [-x.length, x.length)
 * @exception ValueError if x or index is null or undefined
 */
export function pyNormalizeArrayIndex<T>(x: T[], index: number): number {
  if (x == null || index == null) {
    throw new ValueError(
        `Must provide a valid array and index for ` +
        `pyNormalizeArrayIndex(). Got array ${x} and index ${index}.`);
  }
  const errMsg = `Index ${index} out of range for array of length ${x.length}`;
  if (index < 0) {
    if (index < -x.length) {
      throw new IndexError(errMsg);
    }
    return x.length + index;
  }
  if (index >= x.length) {
    throw new IndexError(errMsg);
  }
  return index;
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

export class ClassNameMap {
  private static instance: ClassNameMap;
  pythonClassNameMap: {
    [className: string]:
        [Constructor<Serializable>, FromConfigMethod<Serializable>]
  };

  private constructor() {
    this.pythonClassNameMap = {};
  }

  static getMap() {
    if (ClassNameMap.instance == null) {
      ClassNameMap.instance = new ClassNameMap();
    }
    return ClassNameMap.instance;
  }

  static register<T extends Serializable>(cls: Constructor<T>) {
    this.getMap().pythonClassNameMap[cls.className] = [cls, cls.fromConfig];
  }
}

export class SerializableEnumRegistry {
  private static instance: SerializableEnumRegistry;
  // tslint:disable-next-line:no-any
  enumRegistry: {[fieldName: string]: any};

  private constructor() {
    this.enumRegistry = {};
  }

  static getMap() {
    if (SerializableEnumRegistry.instance == null) {
      SerializableEnumRegistry.instance = new SerializableEnumRegistry();
    }
    return SerializableEnumRegistry.instance;
  }

  // tslint:disable-next-line:no-any
  static register(fieldName: string, enumCls: any) {
    if (SerializableEnumRegistry.contains(fieldName)) {
      throw new ValueError(
          `Attempting to register a repeated enum: ${fieldName}`);
    }
    this.getMap().enumRegistry[fieldName] = enumCls;
  }

  static contains(fieldName: string): boolean {
    return fieldName in this.getMap().enumRegistry;
  }

  // tslint:disable-next-line:no-any
  static lookup(fieldName: string, value: string): any {
    return this.getMap().enumRegistry[fieldName][value];
  }

  // tslint:disable-next-line:no-any
  static reverseLookup(fieldName: string, value: any): string {
    const enumMap = this.getMap().enumRegistry[fieldName];
    for (const candidateString in enumMap) {
      if (enumMap[candidateString] === value) {
        return candidateString;
      }
    }
    throw new ValueError(`Could not find serialization string for ${value}`);
  }
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
 * Determine whether the input is an Array of Shapes.
 */
export function isArrayOfShapes(x: Shape|Shape[]): boolean {
  return Array.isArray(x) && Array.isArray(x[0]);
}

/**
 * Special case of normalizing shapes to lists.
 *
 * @param x A shape or list of shapes to normalize into a list of Shapes.
 * @return A list of Shapes.
 */
export function normalizeShapeList(x: Shape|Shape[]): Shape[] {
  if (x.length === 0) {
    return [];
  }
  if (!Array.isArray(x[0])) {
    return [x] as Shape[];
  }
  return x as Shape[];
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

export function serializeKerasObject(instance: Serializable): ConfigDictValue {
  if (instance === null || instance === undefined) {
    return null;
  }
  return {className: instance.getClassName(), config: instance.getConfig()};
}

/**
 * Deserialize a saved Keras Object
 * @param identifier either a string ID or a saved Keras dictionary
 * @param moduleObjects a list of Python class names to object constructors
 * @param customObjects a list of Python class names to object constructors
 * @param printableModuleName debug text for the object being reconstituted
 * @returns a TensorFlow.js Layers object
 */
// tslint:disable:no-any
export function deserializeKerasObject(
    identifier: string|ConfigDict,
    moduleObjects = {} as {[objName: string]: any},
    customObjects = {} as {[objName: string]: any},
    printableModuleName = 'object'): any {
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
        throw new ValueError(`Unknown ${printableModuleName}: ${identifier}`);
      }
    }
    return fn;
  } else {
    // In this case we are dealing with a Keras config dictionary.
    const config = identifier;
    if (config.className == null || config.config == null) {
      throw new ValueError(
          `${printableModuleName}: Improper config format: ` +
          `${JSON.stringify(config)}.\n` +
          `'className' and 'config' must set.`);
    }
    const className = config.className as string;
    let cls, fromConfig;
    if (className in customObjects) {
      [cls, fromConfig] = customObjects.get(className);
    } else if (className in _GLOBAL_CUSTOM_OBJECTS) {
      [cls, fromConfig] = _GLOBAL_CUSTOM_OBJECTS.className;
    } else if (className in moduleObjects) {
      [cls, fromConfig] = moduleObjects[className];
    }
    if (cls == null) {
      throw new ValueError(`Unknown ${printableModuleName}: ${className}`);
    }
    if (fromConfig != null) {
      // Porting notes: Instead of checking to see whether fromConfig accepts
      // customObjects, we create a customObjects dictionary and tack it on to
      // config.config as config.config.customObjects. Objects can use it, if
      // they want.

      // tslint:disable-next-line:no-any
      const customObjectsCombined = {} as {[objName: string]: any};
      for (const key of Object.keys(_GLOBAL_CUSTOM_OBJECTS)) {
        customObjectsCombined[key] = _GLOBAL_CUSTOM_OBJECTS[key];
      }
      for (const key of Object.keys(customObjects)) {
        customObjectsCombined[key] = customObjects[key];
      }
      // Add the customObjects to config
      const nestedConfig = config.config as ConfigDict;
      nestedConfig.customObjects = customObjectsCombined;

      const backupCustomObjects = {..._GLOBAL_CUSTOM_OBJECTS};
      for (const key of Object.keys(customObjects)) {
        _GLOBAL_CUSTOM_OBJECTS[key] = customObjects[key];
      }
      const returnObj = fromConfig(cls, config.config);
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
      const returnObj = new cls(config.config);
      _GLOBAL_CUSTOM_OBJECTS = {...backupCustomObjects};
      return returnObj;
    }
  }
}

/**
 * Helper function to obtain exactly one Tensor.
 * @param xs: A single `Tensor` or an `Array` of `Tensor`s.
 * @return A single `Tensor`. If `xs` is an `Array`, return the first one.
 * @throws ValueError: If `xs` is an `Array` and its length is not 1.
 */
export function getExactlyOneTensor(xs: Tensor|Tensor[]): Tensor {
  let x: Tensor;
  if (Array.isArray(xs)) {
    if (xs.length !== 1) {
      throw new ValueError(`Expected Tensor length to be 1; got ${xs.length}`);
    }
    x = xs[0];
  } else {
    x = xs as Tensor;
  }
  return x;
}

/**
 * Helper function to obtain exactly on instance of Shape.
 *
 * @param shapes Input single `Shape` or Array of `Shape`s.
 * @returns If input is a single `Shape`, return it unchanged. If the input is
 *   an `Array` containing exactly one instance of `Shape`, return the instance.
 *   Otherwise, throw a `ValueError`.
 * @throws ValueError: If input is an `Array` of `Shape`s, and its length is not
 *   1.
 */
export function getExactlyOneShape(shapes: Shape|Shape[]): Shape {
  if (Array.isArray(shapes) && Array.isArray(shapes[0])) {
    if (shapes.length === 1) {
      shapes = shapes as Shape[];
      return shapes[0];
    } else {
      throw new ValueError(`Expected exactly 1 Shape; got ${shapes.length}`);
    }
  } else {
    return shapes as Shape;
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
export function stringToDType(dtype: string): DType {
  switch (dtype) {
    case 'float32':
      return DType.float32;
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
