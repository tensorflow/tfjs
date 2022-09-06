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

import {MathBackendWebGL} from '../backend_webgl';
import {stringNGramsImplCPU} from '../kernel_utils/shared';

export function stringNGrams(args: {
  inputs: StringNGramsInputs,
  backend: MathBackendWebGL,
  attrs: StringNGramsAttrs
}): [TensorInfo, TensorInfo] {
  const {inputs, backend, attrs} = args;
  const {
    separator,
    nGramWidths,
    leftPad,
    rightPad,
    padWidth,
    preserveShortSequences
  } = attrs;
  const {data, dataSplits} = inputs;
  const $data = backend.readSync(data.dataId) as Uint8Array[];
  const $dataSplits = backend.readSync(dataSplits.dataId) as Int32Array;

  const [nGrams, nGramsSplits] = stringNGramsImplCPU(
      $data, $dataSplits, separator, nGramWidths, leftPad, rightPad, padWidth,
      preserveShortSequences);
  return [
    backend.makeTensorInfo([nGrams.length], 'string', nGrams),
    backend.makeTensorInfo(dataSplits.shape, 'int32', nGramsSplits),
  ];
}

export const stringNGramsConfig: KernelConfig = {
  kernelName: StringNGrams,
  backendName: 'webgl',
  kernelFunc: stringNGrams as {} as KernelFunc,
};
