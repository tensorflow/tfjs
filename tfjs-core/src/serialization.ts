/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {assert} from './util';

/**
 * Types to support JSON-esque data structures internally.
 *
 * Internally ConfigDict's use camelCase keys and values where the
 * values are class names to be instantiated.  On the python side, these
 * will be snake_case.  Internally we allow Enums into the values for better
 * type safety, but these need to be converted to raw primitives (usually
 * strings) for round-tripping with python.
 *
 * toConfig returns the TS-friendly representation. model.toJSON() returns
 * the pythonic version as that's the portable format.  If you need to
 * python-ify a non-model level toConfig output, you'll need to use a
 * convertTsToPythonic from serialization_utils in -Layers.
 *
 */
export declare type ConfigDictValue =
    boolean | number | string | null | ConfigDictArray | ConfigDict;
export declare interface ConfigDict {
  [key: string]: ConfigDictValue;
}
export declare interface ConfigDictArray extends Array<ConfigDictValue> {}

/**
 * Type to represent the class-type of Serializable objects.
 *
 * Ie the class prototype with access to the constructor and any
 * static members/methods. Instance methods are not listed here.
 *
 * Source for this idea: https://stackoverflow.com/a/43607255
 */
export declare type SerializableConstructor<T extends Serializable> = {
  // tslint:disable-next-line:no-any
  new (...args: any[]): T; className: string; fromConfig: FromConfigMethod<T>;
};
export declare type FromConfigMethod<T extends Serializable> =
    (cls: SerializableConstructor<T>, config: ConfigDict) => T;

/**
 * Serializable defines the serialization contract.
 *
 * TFJS requires serializable classes to return their className when asked
 * to avoid issues with minification.
 */
export abstract class Serializable {
  /**
   * Return the class name for this class to use in serialization contexts.
   *
   * Generally speaking this will be the same thing that constructor.name
   * would have returned.  However, the class name needs to be robust
   * against minification for serialization/deserialization to work properly.
   *
   * There's also places such as initializers.VarianceScaling, where
   * implementation details between different languages led to different
   * class hierarchies and a non-leaf node is used for serialization purposes.
   */
  getClassName(): string {
    return (this.constructor as SerializableConstructor<Serializable>)
        .className;
  }

  /**
   * Return all the non-weight state needed to serialize this object.
   */
  abstract getConfig(): ConfigDict;

  /**
   * Creates an instance of T from a ConfigDict.
   *
   * This works for most descendants of serializable.  A few need to
   * provide special handling.
   * @param cls A Constructor for the class to instantiate.
   * @param config The Configuration for the object.
   */
  /** @nocollapse */
  static fromConfig<T extends Serializable>(
      cls: SerializableConstructor<T>, config: ConfigDict): T {
    return new cls(config);
  }
}

/**
 * Maps string keys to class constructors.
 *
 * Used during (de)serialization from the cross-language JSON format, which
 * requires the class name in the serialization format matches the class
 * names as used in Python, should it exist.
 */
export class SerializationMap {
  private static instance: SerializationMap;
  classNameMap: {
    [className: string]:
        [SerializableConstructor<Serializable>, FromConfigMethod<Serializable>]
  };

  private constructor() {
    this.classNameMap = {};
  }

  /**
   * Returns the singleton instance of the map.
   */
  static getMap(): SerializationMap {
    if (SerializationMap.instance == null) {
      SerializationMap.instance = new SerializationMap();
    }
    return SerializationMap.instance;
  }

  /**
   * Registers the class as serializable.
   */
  static register<T extends Serializable>(cls: SerializableConstructor<T>) {
    SerializationMap.getMap().classNameMap[cls.className] =
        [cls, cls.fromConfig];
  }
}

/**
 * Register a class with the serialization map of TensorFlow.js.
 *
 * This is often used for registering custom Layers, so they can be
 * serialized and deserialized.
 *
 * Example:
 *
 * ```js
 * class MyCustomLayer extends tf.layers.Layer {
 *   static className = 'MyCustomLayer';
 *
 *   constructor(config) {
 *     super(config);
 *   }
 * }
 * tf.serialization.registerClass(MyCustomLayer);
 * ```
 *
 * @param cls The class to be registered. It must have a public static member
 *   called `className` defined and the value must be a non-empty string.
 *
 * @doc {heading: 'Models', subheading: 'Serialization', ignoreCI: true}
 */
export function registerClass<T extends Serializable>(
    cls: SerializableConstructor<T>) {
  assert(
      cls.className != null,
      () => `Class being registered does not have the static className ` +
          `property defined.`);
  assert(
      typeof cls.className === 'string',
      () => `className is required to be a string, but got type ` +
          typeof cls.className);
  assert(
      cls.className.length > 0,
      () => `Class being registered has an empty-string as its className, ` +
          `which is disallowed.`);

  SerializationMap.register(cls);
}
