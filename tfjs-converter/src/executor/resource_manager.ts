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
import {HashTableMap, NamedTensorMap} from '../data/types';
import {HashTable} from './hash_table';

export class ResourceManager {
  constructor(
      readonly hashTableNameToHandle: NamedTensorMap = {},
      readonly hashTableMap: HashTableMap = {}) {}

  addHashTable(name: string, hashTable: HashTable) {
    this.hashTableNameToHandle[name] = hashTable.handle;
    this.hashTableMap[hashTable.id] = hashTable;
  }

  getHashTableHandleByName(name: string) {
    return this.hashTableNameToHandle[name];
  }

  getHashTableById(id: number): HashTable {
    return this.hashTableMap[id];
  }

  dispose() {
    for (const key in this.hashTableMap) {
      this.hashTableMap[key].clearAndClose();
    }
    for (const name in this.hashTableNameToHandle) {
      this.hashTableNameToHandle[name].dispose();
    }
  }
}
