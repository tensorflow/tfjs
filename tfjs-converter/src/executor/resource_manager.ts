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
import {HashTableMap, NamedTensorMap} from '../data/types';
import {HashTable} from './hash_table';

/**
 * Contains global resources of a model.
 */
export class ResourceManager {
  constructor(
      readonly hashTableNameToHandle: NamedTensorMap = {},
      readonly hashTableMap: HashTableMap = {}) {}

  /**
   * Register a `HashTable` in the resource manager.
   *
   * The `HashTable` can be retrieved by `resourceManager.getHashTableById`,
   * where id is the table handle tensor's id.
   *
   * @param name Op node name that creates the `HashTable`.
   * @param hashTable The `HashTable` to be added to resource manager.
   */
  addHashTable(name: string, hashTable: HashTable) {
    this.hashTableNameToHandle[name] = hashTable.handle;
    this.hashTableMap[hashTable.id] = hashTable;
  }

  /**
   * Get the table handle by node name.
   * @param name Op node name that creates the `HashTable`. This name is also
   *     used in the inputs list of lookup and import `HashTable` ops.
   */
  getHashTableHandleByName(name: string) {
    return this.hashTableNameToHandle[name];
  }

  /**
   * Get the actual `HashTable` by its handle tensor's id.
   * @param id The id of the handle tensor.
   */
  getHashTableById(id: number): HashTable {
    return this.hashTableMap[id];
  }

  /**
   * Dispose `ResourceManager`, including its hashTables and tensors in them.
   */
  dispose() {
    for (const key in this.hashTableMap) {
      this.hashTableMap[key].clearAndClose();
      delete this.hashTableMap[key];
    }

    for (const name in this.hashTableNameToHandle) {
      this.hashTableNameToHandle[name].dispose();
      delete this.hashTableNameToHandle[name];
    }
  }
}
