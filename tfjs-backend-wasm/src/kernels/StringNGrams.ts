/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {KernelConfig, KernelFunc, StringNGrams, StringNGramsAttrs, StringNGramsInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';
import {stringNGramsImplCPU} from '../kernel_utils/shared';

function stringNGrams(args: {
  backend: BackendWasm,
  inputs: StringNGramsInputs,
  attrs: StringNGramsAttrs
}): [TensorInfo, TensorInfo] {
  const {backend, inputs, attrs} = args;
  const {data, dataSplits} = inputs;
  const {
    separator,
    nGramWidths,
    leftPad,
    rightPad,
    padWidth,
    preserveShortSequences,
  } = attrs;

  const $data = backend.readSync(data.dataId) as Uint8Array[];
  const $dataSplits = backend.readSync(dataSplits.dataId) as Int32Array;

  const [nGrams, nGramsSplits] = stringNGramsImplCPU(
      $data, $dataSplits, separator, nGramWidths, leftPad, rightPad, padWidth,
      preserveShortSequences);

  const nGramsOut = backend.makeOutput([nGrams.length], 'string');
  const nGramsOutData = backend.dataIdMap.get(nGramsOut.dataId);
  nGramsOutData.stringBytes = nGrams;

  const nGramsSplitsOut = backend.makeOutput(dataSplits.shape, 'int32');
  const nGramsSplitsOutVals = backend.typedArrayFromHeap(nGramsSplitsOut);
  nGramsSplitsOutVals.set(nGramsSplits);

  return [nGramsOut, nGramsSplitsOut];
}

export const stringNGramsConfig: KernelConfig = {
  kernelName: StringNGrams,
  backendName: 'wasm',
  kernelFunc: stringNGrams as unknown as KernelFunc
};
