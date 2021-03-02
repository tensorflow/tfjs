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
 *
 * =============================================================================
 */
import {memory, test_util} from '@tensorflow/tfjs-core';

// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {ExecutionContext} from '../../executor/execution_context';
import {HashTable} from '../../executor/hash_table';
import {ResourceManager} from '../../executor/resource_manager';
import * as hashTable from '../op_list/hash_table';
import {Node} from '../types';

import {executeOp} from './hash_table_executor';
import {createDtypeAttr, createTensorAttr, validateParam} from './test_helper';

describe('hash_table', () => {
  let node: Node;
  const context = new ExecutionContext({}, {}, {});
  let resourceManager: ResourceManager;

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'hash_table',
      inputNames: [],
      inputs: [],
      inputParams: {},
      attrParams: {},
      children: []
    };
    resourceManager = new ResourceManager();
  });

  afterEach(() => {
    resourceManager.dispose();
  });

  describe('executeOp', () => {
    describe('HashTable', () => {
      it('should create new tensor on the resourceManager.', async () => {
        node.op = 'HashTable';
        node.attrParams['keyDType'] = createDtypeAttr('string');
        node.attrParams['valueDType'] = createDtypeAttr('float32');

        const before = memory().numTensors;
        const handle = (await executeOp(node, {}, context, resourceManager))[0];
        const after = memory().numTensors;
        // 1 handle is created.
        expect(after).toBe(before + 1);
        expect(resourceManager.getHashTableById(handle.id)).toBeDefined();
        expect(Object.keys(resourceManager.hashTableMap).length).toBe(1);
      });
      it('should match json def.', () => {
        node.op = 'HashTable';
        node.attrParams['keyDType'] = createDtypeAttr('string');
        node.attrParams['valueDType'] = createDtypeAttr('float32');

        expect(validateParam(node, hashTable.json)).toBeTruthy();
      });
    });

    describe('HashTableV2', () => {
      it('should create new tensor on the resourceManager.', async () => {
        node.op = 'HashTableV2';
        node.attrParams['keyDType'] = createDtypeAttr('string');
        node.attrParams['valueDType'] = createDtypeAttr('float32');

        const before = memory().numTensors;
        const handle = (await executeOp(node, {}, context, resourceManager))[0];
        const after = memory().numTensors;
        // 1 handle is created.
        expect(after).toBe(before + 1);
        expect(resourceManager.getHashTableById(handle.id)).toBeDefined();
        expect(Object.keys(resourceManager.hashTableMap).length).toBe(1);
      });
      it('should match json def.', () => {
        node.op = 'HashTableV2';
        node.attrParams['keyDType'] = createDtypeAttr('string');
        node.attrParams['valueDType'] = createDtypeAttr('float32');

        expect(validateParam(node, hashTable.json)).toBeTruthy();
      });
    });

    describe('LookupTableImport', () => {
      it('should return hashTable handle.', async () => {
        const hashTable = new HashTable('string', 'float32');

        resourceManager.addHashTable('hashtable', hashTable);

        node.op = 'LookupTableImport';
        node.inputParams['tableHandle'] = createTensorAttr(0);
        node.inputParams['keys'] = createTensorAttr(1);
        node.inputParams['values'] = createTensorAttr(2);
        node.inputNames = ['hashtable', 'input3', 'input5'];
        const input3 = [tfOps.tensor1d(['a'], 'string')];
        const input5 = [tfOps.tensor1d([5.5])];

        const before = memory().numTensors;
        (await executeOp(node, {input3, input5}, context, resourceManager));
        const after = memory().numTensors;
        expect(after).toBe(before + 1);
      });

      it('should throw if dtype doesnot match.', async (done) => {
        const hashTable = new HashTable('string', 'float32');

        resourceManager.addHashTable('hashtable', hashTable);

        node.op = 'LookupTableImport';
        node.inputParams['tableHandle'] = createTensorAttr(0);
        node.inputParams['keys'] = createTensorAttr(1);
        node.inputParams['values'] = createTensorAttr(2);
        node.inputNames = ['hashtable', 'input3', 'input5'];
        const input3 = [tfOps.tensor1d([1])];
        const input5 = [tfOps.tensor1d([5.5])];

        const before = memory().numTensors;
        try {
          await executeOp(node, {input3, input5}, context, resourceManager);
          done.fail('Should fail, succeed unexpectedly.');
        } catch (err) {
          expect(err).toMatch(/Expect key dtype/);
        }
        const after = memory().numTensors;
        expect(after).toBe(before);
        done();
      });

      it('should throw if length of keys and values doesnot match.',
         async (done) => {
           const hashTable = new HashTable('string', 'float32');

           resourceManager.addHashTable('hashtable', hashTable);

           node.op = 'LookupTableImport';
           node.inputParams['tableHandle'] = createTensorAttr(0);
           node.inputParams['keys'] = createTensorAttr(1);
           node.inputParams['values'] = createTensorAttr(2);
           node.inputNames = ['hashtable', 'input3', 'input5'];
           const input3 = [tfOps.tensor1d(['a', 'b'])];
           const input5 = [tfOps.tensor1d([5.5])];

           const before = memory().numTensors;
           try {
             await executeOp(node, {input3, input5}, context, resourceManager);
             done.fail('Should fail, succeed unexpectedly.');
           } catch (err) {
             expect(err).toMatch(/The number of elements doesn't match/);
           }
           const after = memory().numTensors;
           expect(after).toBe(before);
           done();
         });

      it('should match json def.', () => {
        node.op = 'LookupTableImport';
        node.inputParams['tableHandle'] = createTensorAttr(0);
        node.inputParams['keys'] = createTensorAttr(1);
        node.inputParams['values'] = createTensorAttr(2);

        expect(validateParam(node, hashTable.json)).toBeTruthy();
      });
    });

    describe('LookupTableImportV2', () => {
      it('should return hashTable handle.', async () => {
        const hashTable = new HashTable('string', 'float32');

        resourceManager.addHashTable('hashtable', hashTable);

        node.op = 'LookupTableImportV2';
        node.inputParams['tableHandle'] = createTensorAttr(0);
        node.inputParams['keys'] = createTensorAttr(1);
        node.inputParams['values'] = createTensorAttr(2);
        node.inputNames = ['hashtable', 'input3', 'input5'];
        const input3 = [tfOps.tensor1d(['a'], 'string')];
        const input5 = [tfOps.tensor1d([5.5])];

        const before = memory().numTensors;
        (await executeOp(node, {input3, input5}, context, resourceManager));
        const after = memory().numTensors;
        expect(after).toBe(before + 1);
      });

      it('should throw if dtype doesnot match.', async (done) => {
        const hashTable = new HashTable('string', 'float32');

        resourceManager.addHashTable('hashtable', hashTable);

        node.op = 'LookupTableImportV2';
        node.inputParams['tableHandle'] = createTensorAttr(0);
        node.inputParams['keys'] = createTensorAttr(1);
        node.inputParams['values'] = createTensorAttr(2);
        node.inputNames = ['hashtable', 'input3', 'input5'];
        const input3 = [tfOps.tensor1d([1])];
        const input5 = [tfOps.tensor1d([5.5])];

        const before = memory().numTensors;
        try {
          await executeOp(node, {input3, input5}, context, resourceManager);
          done.fail('Should fail, succeed unexpectedly.');
        } catch (err) {
          expect(err).toMatch(/Expect key dtype/);
        }
        const after = memory().numTensors;
        expect(after).toBe(before);
        done();
      });

      it('should throw if length of keys and values doesnot match.',
         async (done) => {
           const hashTable = new HashTable('string', 'float32');

           resourceManager.addHashTable('hashtable', hashTable);

           node.op = 'LookupTableImportV2';
           node.inputParams['tableHandle'] = createTensorAttr(0);
           node.inputParams['keys'] = createTensorAttr(1);
           node.inputParams['values'] = createTensorAttr(2);
           node.inputNames = ['hashtable', 'input3', 'input5'];
           const input3 = [tfOps.tensor1d(['a', 'b'])];
           const input5 = [tfOps.tensor1d([5.5])];

           const before = memory().numTensors;
           try {
             await executeOp(node, {input3, input5}, context, resourceManager);
             done.fail('Should fail, succeed unexpectedly.');
           } catch (err) {
             expect(err).toMatch(/The number of elements doesn't match/);
           }
           const after = memory().numTensors;
           expect(after).toBe(before);
           done();
         });

      it('should match json def.', () => {
        node.op = 'LookupTableImportV2';
        node.inputParams['tableHandle'] = createTensorAttr(0);
        node.inputParams['keys'] = createTensorAttr(1);
        node.inputParams['values'] = createTensorAttr(2);

        expect(validateParam(node, hashTable.json)).toBeTruthy();
      });
    });

    describe('LookupTableFind', () => {
      it('should find the value from hashtable.', async () => {
        const hashTable = new HashTable('string', 'float32');

        resourceManager.addHashTable('hashtable', hashTable);

        const importNode: Node = {
          name: 'importv2',
          op: 'LookupTableImportV2',
          category: 'hash_table',
          inputNames: ['hashtable', 'keys', 'values'],
          inputs: [],
          inputParams: {
            tableHandle: createTensorAttr(0),
            keys: createTensorAttr(1),
            values: createTensorAttr(2)
          },
          attrParams: {},
          children: []
        };

        const keys = [tfOps.tensor1d(['a'], 'string')];
        const values = [tfOps.tensor1d([5.5])];

        (await executeOp(importNode, {keys, values}, context, resourceManager));

        node.op = 'LookupTableFind';
        node.inputParams['tableHandle'] = createTensorAttr(0);
        node.inputParams['keys'] = createTensorAttr(1);
        node.inputParams['defaultValue'] = createTensorAttr(2);
        node.inputNames = ['hashtable', 'input3', 'input5'];
        const input3 = [tfOps.tensor1d(['a', 'b'], 'string')];
        const input5 = [tfOps.scalar(0)];

        const before = memory().numTensors;

        const result =
            (await executeOp(node, {input3, input5}, context, resourceManager));
        const after = memory().numTensors;
        test_util.expectArraysClose(await result[0].data(), [5.5, 0]);

        // Create a result tensor.
        expect(after).toBe(before + 1);
      });
      it('should throw if dtype doesnot match.', async (done) => {
        const hashTable = new HashTable('string', 'float32');

        resourceManager.addHashTable('hashtable', hashTable);

        const importNode: Node = {
          name: 'importv2',
          op: 'LookupTableImportV2',
          category: 'hash_table',
          inputNames: ['hashtable', 'keys', 'values'],
          inputs: [],
          inputParams: {
            tableHandle: createTensorAttr(0),
            keys: createTensorAttr(1),
            values: createTensorAttr(2)
          },
          attrParams: {},
          children: []
        };

        const keys = [tfOps.tensor1d(['a'], 'string')];
        const values = [tfOps.tensor1d([5.5])];

        (await executeOp(importNode, {keys, values}, context, resourceManager));

        node.op = 'LookupTableFind';
        node.inputParams['tableHandle'] = createTensorAttr(0);
        node.inputParams['keys'] = createTensorAttr(1);
        node.inputParams['defaultValue'] = createTensorAttr(2);
        node.inputNames = ['hashtable', 'input3', 'input5'];
        const input3 = [tfOps.tensor1d([1, 2], 'float32')];
        const input5 = [tfOps.scalar(0)];

        const before = memory().numTensors;
        try {
          await executeOp(node, {input3, input5}, context, resourceManager);
          done.fail('Shoudl fail, succeed unexpectedly.');
        } catch (err) {
          expect(err).toMatch(/Expect key dtype/);
        }
        const after = memory().numTensors;
        expect(after).toBe(before);
        done();
      });
      it('should match json def.', () => {
        node.op = 'LookupTableFind';
        node.inputParams['tableHandle'] = createTensorAttr(0);
        node.inputParams['keys'] = createTensorAttr(1);
        node.inputParams['defaultValue'] = createTensorAttr(2);

        expect(validateParam(node, hashTable.json)).toBeTruthy();
      });
    });

    describe('LookupTableFindV2', () => {
      it('should find the value from hashtable.', async () => {
        const hashTable = new HashTable('string', 'float32');

        resourceManager.addHashTable('hashtable', hashTable);

        const importNode: Node = {
          name: 'importv2',
          op: 'LookupTableImportV2',
          category: 'hash_table',
          inputNames: ['hashtable', 'keys', 'values'],
          inputs: [],
          inputParams: {
            tableHandle: createTensorAttr(0),
            keys: createTensorAttr(1),
            values: createTensorAttr(2)
          },
          attrParams: {},
          children: []
        };

        const keys = [tfOps.tensor1d(['a'], 'string')];
        const values = [tfOps.tensor1d([5.5])];

        (await executeOp(importNode, {keys, values}, context, resourceManager));

        node.op = 'LookupTableFindV2';
        node.inputParams['tableHandle'] = createTensorAttr(0);
        node.inputParams['keys'] = createTensorAttr(1);
        node.inputParams['defaultValue'] = createTensorAttr(2);
        node.inputNames = ['hashtable', 'input3', 'input5'];
        const input3 = [tfOps.tensor1d(['a', 'b'], 'string')];
        const input5 = [tfOps.scalar(0)];

        const before = memory().numTensors;

        const result =
            (await executeOp(node, {input3, input5}, context, resourceManager));
        const after = memory().numTensors;
        test_util.expectArraysClose(await result[0].data(), [5.5, 0]);

        // Create a result tensor.
        expect(after).toBe(before + 1);
      });
      it('should throw if dtype doesnot match.', async (done) => {
        const hashTable = new HashTable('string', 'float32');

        resourceManager.addHashTable('hashtable', hashTable);

        const importNode: Node = {
          name: 'importv2',
          op: 'LookupTableImportV2',
          category: 'hash_table',
          inputNames: ['hashtable', 'keys', 'values'],
          inputs: [],
          inputParams: {
            tableHandle: createTensorAttr(0),
            keys: createTensorAttr(1),
            values: createTensorAttr(2)
          },
          attrParams: {},
          children: []
        };

        const keys = [tfOps.tensor1d(['a'], 'string')];
        const values = [tfOps.tensor1d([5.5])];

        (await executeOp(importNode, {keys, values}, context, resourceManager));

        node.op = 'LookupTableFindV2';
        node.inputParams['tableHandle'] = createTensorAttr(0);
        node.inputParams['keys'] = createTensorAttr(1);
        node.inputParams['defaultValue'] = createTensorAttr(2);
        node.inputNames = ['hashtable', 'input3', 'input5'];
        const input3 = [tfOps.tensor1d([1, 2], 'float32')];
        const input5 = [tfOps.scalar(0)];

        const before = memory().numTensors;
        try {
          await executeOp(node, {input3, input5}, context, resourceManager);
          done.fail('Shoudl fail, succeed unexpectedly.');
        } catch (err) {
          expect(err).toMatch(/Expect key dtype/);
        }
        const after = memory().numTensors;
        expect(after).toBe(before);
        done();
      });
      it('should match json def.', () => {
        node.op = 'LookupTableFindV2';
        node.inputParams['tableHandle'] = createTensorAttr(0);
        node.inputParams['keys'] = createTensorAttr(1);
        node.inputParams['defaultValue'] = createTensorAttr(2);

        expect(validateParam(node, hashTable.json)).toBeTruthy();
      });
    });
    describe('LookupTableSize', () => {
      it('should return 0 if the hashtable is empty.', async () => {
        const hashTable = new HashTable('string', 'float32');
        resourceManager.addHashTable('hashtablesize', hashTable);
        const importNode: Node = {
          name: 'importv2',
          op: 'LookupTableImportV2',
          category: 'hash_table',
          inputNames: ['hashtablesize', 'keys', 'values'],
          inputs: [],
          inputParams: {
            tableHandle: createTensorAttr(0),
            keys: createTensorAttr(1),
            values: createTensorAttr(2)
          },
          attrParams: {},
          children: []
        };
        // Import empty tensors
        const keys = [tfOps.tensor1d([], 'string')];
        const values = [tfOps.tensor1d([])];
        (await executeOp(importNode, {keys, values}, context, resourceManager));

        node.op = 'LookupTableSize';
        node.inputParams['tableHandle'] = createTensorAttr(0);
        node.inputNames = ['hashtablesize'];
        const before = memory().numTensors;
        const result = await executeOp(node, {}, context, resourceManager);
        const after = memory().numTensors;
        const size = await result[0].data();

        test_util.expectArraysClose(size, [0]);
        expect(after).toBe(before + 1);
      });
      it('should return the number of elements in the hashtable.', async () => {
        const hashTable = new HashTable('string', 'float32');
        resourceManager.addHashTable('hashtablesize', hashTable);
        const importNode: Node = {
          name: 'importv2',
          op: 'LookupTableImportV2',
          category: 'hash_table',
          inputNames: ['hashtablesize', 'keys', 'values'],
          inputs: [],
          inputParams: {
            tableHandle: createTensorAttr(0),
            keys: createTensorAttr(1),
            values: createTensorAttr(2)
          },
          attrParams: {},
          children: []
        };
        // Import tensors with 2 elements
        const keys = [tfOps.tensor1d(['a', 'b'], 'string')];
        const values = [tfOps.tensor1d([5.5, 10])];
        (await executeOp(importNode, {keys, values}, context, resourceManager));

        node.op = 'LookupTableSize';
        node.inputParams['tableHandle'] = createTensorAttr(0);
        node.inputNames = ['hashtablesize'];
        const before = memory().numTensors;
        const result = await executeOp(node, {}, context, resourceManager);
        const after = memory().numTensors;
        const size = await result[0].data();

        test_util.expectArraysClose(size, [2]);
        expect(after).toBe(before + 1);
      });
    });
    describe('LookupTableSizeV2', () => {
      it('should return 0 if the hashtable is empty.', async () => {
        const hashTable = new HashTable('string', 'float32');
        resourceManager.addHashTable('hashtablesize', hashTable);
        const importNode: Node = {
          name: 'importv2',
          op: 'LookupTableImportV2',
          category: 'hash_table',
          inputNames: ['hashtablesize', 'keys', 'values'],
          inputs: [],
          inputParams: {
            tableHandle: createTensorAttr(0),
            keys: createTensorAttr(1),
            values: createTensorAttr(2)
          },
          attrParams: {},
          children: []
        };
        // Import empty tensors
        const keys = [tfOps.tensor1d([], 'string')];
        const values = [tfOps.tensor1d([])];
        (await executeOp(importNode, {keys, values}, context, resourceManager));

        node.op = 'LookupTableSizeV2';
        node.inputParams['tableHandle'] = createTensorAttr(0);
        node.inputNames = ['hashtablesize'];
        const before = memory().numTensors;
        const result = await executeOp(node, {}, context, resourceManager);
        const after = memory().numTensors;
        const size = await result[0].data();

        test_util.expectArraysClose(size, [0]);
        expect(after).toBe(before + 1);
      });
      it('should return the number of elements in the hashtable.', async () => {
        const hashTable = new HashTable('string', 'float32');
        resourceManager.addHashTable('hashtablesize', hashTable);
        const importNode: Node = {
          name: 'importv2',
          op: 'LookupTableImportV2',
          category: 'hash_table',
          inputNames: ['hashtablesize', 'keys', 'values'],
          inputs: [],
          inputParams: {
            tableHandle: createTensorAttr(0),
            keys: createTensorAttr(1),
            values: createTensorAttr(2)
          },
          attrParams: {},
          children: []
        };
        // Import tensors with 2 elements
        const keys = [tfOps.tensor1d(['a', 'b'], 'string')];
        const values = [tfOps.tensor1d([5.5, 10])];
        (await executeOp(importNode, {keys, values}, context, resourceManager));

        node.op = 'LookupTableSizeV2';
        node.inputParams['tableHandle'] = createTensorAttr(0);
        node.inputNames = ['hashtablesize'];
        const before = memory().numTensors;
        const result = await executeOp(node, {}, context, resourceManager);
        const after = memory().numTensors;
        const size = await result[0].data();

        test_util.expectArraysClose(size, [2]);
        expect(after).toBe(before + 1);
      });
    });
  });
});
