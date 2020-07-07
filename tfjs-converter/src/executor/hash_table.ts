/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import {DataType, equal, keep, scalar, stack, Tensor, unstack, util} from '@tensorflow/tfjs-core';

/**
 * Hashtable contains a set of tensors, which can be accessed by key.
 */
export class HashTable {
  readonly handle: Tensor;

  private tensorMap: Map<Tensor, Tensor>;
  private initialized_ = false;

  /**
   * Constructor of HashTable. Creates a non-initialized hash table.
   *
   * @param keyDType `dtype` of the table keys.
   * @param valueDType `dtype` of the table values.
   * @param sharedName Optional. Defaults to ''. The name used to share the
   *     table.
   * @param useNodeNameSharing Optional. Defaults to false. If true and
   *     sharedName is empty, the table is shared using the node name.
   * @param name Optional. Name of the node.
   */
  constructor(
      readonly keyDType: DataType, readonly valueDType: DataType,
      readonly sharedName: string = '',
      readonly useNodeNameSharing: boolean = false,
      readonly name: string = '') {
    if (useNodeNameSharing && sharedName === '') {
      this.handle = scalar(util.encodeString(name), 'string');
    }

    this.handle = scalar(util.encodeString(sharedName), 'string');

    keep(this.handle);
  }

  get initialized(): boolean {
    return this.initialized_;
  }

  /**
   * Dispose the tensors and idTensor and clear the hashtable.
   */
  clearAndClose() {
    this.tensorMap.clear();
    this.handle.dispose();
  }

  /**
   * The number of items in the hash table.
   */
  size(): number {
    return this.tensorMap.size;
  }

  /**
   * Looks up keys in a hash table, outputs the corresponding values.
   *
   * Performs batch lookups, for every element in the key tensor, `find`
   * stacks the corresponding value into the return tensor. Each lookup takes
   * O(n) time, n is the size of the table.
   *
   * If an element is not present in the table, the given `defaultValue` is
   * used.
   *
   * @param keys Keys to look up. Must have the same type as the keys of the
   *     table.
   * @param defaultValue The scalar `defaultValue` is the value output for keys
   *     not present in the table. It must also be of the same type as the
   *     table values.
   */
  async find(keys: Tensor, defaultValue: Tensor): Promise<Tensor> {
    const $keys = unstack(keys);

    const result: Tensor[] = [];
    for (let i = 0; i < $keys.length; i++) {
      const key = $keys[i];
      this.checkKeyAndValueTensor(key, defaultValue);

      const value = await this.findWithDefault(key, defaultValue);

      result.push(value);
    }

    return stack(result);
  }

  /**
   * Initializes the table with associated keys and values.
   * @param keys Keys to store in the hashtable.
   * @param values Values to store in the hashtable.
   */
  initialize(keys: Tensor, values: Tensor) {
    if (this.initialized_) {
      throw new Error(`Hashtable ${this.sharedName} already initialized.`);
    }

    this.tensorMap = new Map<Tensor, Tensor>();
    this.insert(keys, values);
    this.initialized_ = true;
  }

  private insert(keys: Tensor, values: Tensor) {
    const $keys = unstack(keys);
    const $values = unstack(values);

    for (let i = 0; i < $keys.length; i++) {
      const key = $keys[i];
      const value = $values[i];

      this.checkKeyAndValueTensor(key, value);

      if (this.tensorMap.has(key) && this.tensorMap.get(key) !== value) {
        throw new Error(`HashTable ${this.sharedName} already has key ${key}`);
      }

      this.tensorMap.set(key, value);
    }
  }

  private async findWithDefault(keyToFind: Tensor, defaultValue: Tensor):
      Promise<Tensor> {
    for (const [key, value] of this.tensorMap) {
      const $isEqual = (await equal(key, keyToFind).data())[0];
      if ($isEqual) {
        return value;
      }
    }

    return defaultValue;
  }

  private checkKeyAndValueTensor(key: Tensor, value: Tensor) {
    if (key.dtype !== this.keyDType) {
      throw new Error(
          `Expect key dtype ${this.keyDType}, but got ` +
          `${key.dtype}`);
    }

    if (value.dtype !== this.valueDType) {
      throw new Error(
          `Expect value dtype ${this.valueDType}, but got ` +
          `${value.dtype}`);
    }
  }
}
