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

import {NamedTensorInfoMap, registerKernel, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {batchMatMul} from './BatchMatMul';
import {generateKernelFunc as generateBinaryKernelFunc} from './binary_kernel';
import {concat} from './Concat';
import {slice} from './Slice';

interface LSTMInputs extends NamedTensorInfoMap {
  forgetBias: TensorInfo;
  lstmKernel: TensorInfo;
  lstmBias: TensorInfo;
  data: TensorInfo;
  c: TensorInfo;
  h: TensorInfo;
}

let wasmFunc: (
    iId: number, jId: number, cId: number, forgetBiasId: number, fId: number,
    oId: number, newCId: number, newHId: number) => void;

function setup(backend: BackendWasm): void {
  wasmFunc = backend.wasm.cwrap('LSTM', null /* void */, [
    'number', 'number', 'number', 'number', 'number', 'number', 'number',
    'number'
  ]);
}

function lstm(args: {backend: BackendWasm, inputs: LSTMInputs, attrs: {}}):
    [TensorInfo, TensorInfo] {
  // wasmFunc();
  const {backend, inputs: {forgetBias, lstmKernel, lstmBias, data, c, h}} =
      args;
  const forgetBiasId = backend.dataIdMap.get(forgetBias.dataId).id;
  // const lstmKernelId = backend.dataIdMap.get(lstmKernel.dataId).id;
  // const lstmBiasId = backend.dataIdMap.get(lstmBias.dataId).id;
  // const dataId = backend.dataIdMap.get(data.dataId).id;
  const cId = backend.dataIdMap.get(c.dataId).id;
  // const hId = backend.dataIdMap.get(h.dataId).id;

  const combined = concat({inputs: [data, h], attrs: {axis: 1}, backend});
  const weighted = batchMatMul({
    inputs: {a: combined, b: lstmKernel},
    attrs: {transposeA: false, transposeB: false},
    backend
  });
  const addKernel = generateBinaryKernelFunc('Add', weighted.dtype, true, {
    wasmFunc:
        (aId: number, aShape: Uint8Array, aShapeLen: number, bId: number,
         bShape: Uint8Array, bShapeLen: number, dtype: number, outId: number):
            any => {}
  });
  const res = addKernel({inputs: {a: weighted, b: lstmBias}, backend});

  const batchSize = res.shape[0];
  const sliceCols = res.shape[1] / 4;
  const sliceSize: [number, number] = [batchSize, sliceCols];

  const i = slice(
      {inputs: {x: res}, attrs: {begin: [0, 0], size: sliceSize}, backend});
  const iId = backend.dataIdMap.get(i.dataId).id;
  const j = slice({
    inputs: {x: res},
    attrs: {begin: [0, sliceCols], size: sliceSize},
    backend
  });
  const jId = backend.dataIdMap.get(j.dataId).id;
  const f = slice({
    inputs: {x: res},
    attrs: {begin: [0, sliceCols * 2], size: sliceSize},
    backend
  });
  const fId = backend.dataIdMap.get(f.dataId).id;
  const o = slice({
    inputs: {x: res},
    attrs: {begin: [0, sliceCols * 3], size: sliceSize},
    backend
  });
  const oId = backend.dataIdMap.get(o.dataId).id;

  const newC = backend.makeOutput(i.shape, i.dtype);
  const newCId = backend.dataIdMap.get(newC.dataId).id;
  const newH = backend.makeOutput(newC.shape, newC.dtype);
  const newHId = backend.dataIdMap.get(newH.dataId).id;

  wasmFunc(iId, jId, cId, forgetBiasId, fId, oId, newCId, newHId);

  return [newC, newH];
}

registerKernel({
  kernelName: 'LSTM',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: lstm
});
