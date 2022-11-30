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

import {KernelConfig, KernelFunc, StringSplit, StringSplitAttrs, StringSplitInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';
import {stringSplitImplCPU} from '../kernel_utils/shared';

function stringSplit(args: {
  backend: BackendWasm,
  inputs: StringSplitInputs,
  attrs: StringSplitAttrs
}): [TensorInfo, TensorInfo, TensorInfo] {
  const {backend, inputs, attrs} = args;
  const {input, delimiter} = inputs;
  const {skipEmpty} = attrs;

  const inputVals = backend.readSync(input.dataId) as Uint8Array[];
  const delimiterVals = backend.readSync(delimiter.dataId) as Uint8Array[];

  const [indices, values, shape] =
      stringSplitImplCPU(inputVals, delimiterVals[0], skipEmpty);
  const outputSize = values.length;

  const indicesOut = backend.makeOutput([outputSize, 2], 'int32');
  const indicesOutVals = backend.typedArrayFromHeap(indicesOut);
  indicesOutVals.set(indices);

  const valuesOut = backend.makeOutput([outputSize], 'string');
  const valuesOutData = backend.dataIdMap.get(valuesOut.dataId);
  valuesOutData.stringBytes = values;

  const shapeOut = backend.makeOutput([2], 'int32');
  const shapeOutVals = backend.typedArrayFromHeap(shapeOut);
  shapeOutVals.set(shape);

  return [indicesOut, valuesOut, shapeOut];
}

export const stringSplitConfig: KernelConfig = {
  kernelName: StringSplit,
  backendName: 'wasm',
  kernelFunc: stringSplit as unknown as KernelFunc
};
