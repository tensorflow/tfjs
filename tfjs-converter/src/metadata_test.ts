/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

// This test checks the metadata.json to make sure that only the kernels that
// we know we don't map to tfjs ops have empty entries in metadata
// kernel2op.json
describe('kernel2op metadata file', () => {
  it('has kernel2op.json', () => {
    expect(() => {
      // tslint:disable-next-line:no-require-imports
      require('../metadata/kernel2op.json');
    }).not.toThrow();
  });

  it('only known unmapped kernel are unmmapped', () => {
    const knownUnmappedKernels = [
      'Const',
      'EmptyTensorList',
      'Enter',
      'Exit',
      'FakeQuantWithMinMaxVars',
      'Identity',
      'IdentityN',
      'If',
      'LoopCond',
      'Merge',
      'NextIteration',
      'Placeholder',
      'PlaceholderWithDefault',
      'Print',
      'Snapshot',
      'StatelessIf',
      'StatelessWhile',
      'StopGradient',
      'Switch',
      'TensorArrayCloseV3',
      'TensorArrayConcatV3',
      'TensorArrayGatherV3',
      'TensorArrayReadV3',
      'TensorArrayScatterV3',
      'TensorArraySizeV3',
      'TensorArraySplitV3',
      'TensorArrayV3',
      'TensorArrayWriteV3',
      'TensorListConcat',
      'TensorListFromTensor',
      'TensorListGather',
      'TensorListGetItem',
      'TensorListPopBack',
      'TensorListPushBack',
      'TensorListReserve',
      'TensorListScatter',
      'TensorListScatterV2',
      'TensorListSetItem',
      'TensorListSplit',
      'TensorListStack',
      'While',
      'HashTable',
      'HashTableV2',
      'LookupTableImport',
      'LookupTableImportV2',
      'LookupTableFind',
      'LookupTableFindV2',
      'LookupTableSize',
      'LookupTableSizeV2'
    ];
    // tslint:disable-next-line:no-require-imports
    const kernel2op = require('../metadata/kernel2op.json');
    const kernels: string[] = Object.keys(kernel2op);

    for (const kernelName of kernels) {
      const tfOps = kernel2op[kernelName];
      if (knownUnmappedKernels.includes(kernelName)) {
        expect(tfOps.length)
            .toEqual(0, `Kernel "${kernelName}" is expected to be unmapped but
              instead maps to ${tfOps}`);
      } else {
        expect(tfOps.length)
            .toBeGreaterThan(
                0, `Kernel ${kernelName} is expected to be mapped to a list
                of tf ops`);
      }
    }
  });
});
