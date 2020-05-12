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
(function (global, factory) {
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('@tensorflow/tfjs-core'), require('../wasm-out/tfjs-backend-wasm.js')) :
  typeof define === 'function' && define.amd ? define(['exports', '@tensorflow/tfjs-core', '../wasm-out/tfjs-backend-wasm.js'], factory) :
  (global = global || self, factory((global.tf = global.tf || {}, global.tf.wasm = global.tf.wasm || {}), global.tf, global.WasmBackendModule));
}(this, (function (exports, tfjsCore, WasmBackendModule) { 'use strict';

  WasmBackendModule = WasmBackendModule && WasmBackendModule.hasOwnProperty('default') ? WasmBackendModule['default'] : WasmBackendModule;

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  // This enum must align with the enum defined in cc/backend.h.
  var CppDType;
  (function (CppDType) {
      CppDType[CppDType["float32"] = 0] = "float32";
      CppDType[CppDType["int32"] = 1] = "int32";
      CppDType[CppDType["bool"] = 2] = "bool";
      CppDType[CppDType["string"] = 3] = "string";
      CppDType[CppDType["complex64"] = 4] = "complex64";
  })(CppDType || (CppDType = {}));
  // Must match enum in cc/fusable_activations.h.
  var FusableActivation;
  (function (FusableActivation) {
      FusableActivation[FusableActivation["linear"] = 0] = "linear";
      FusableActivation[FusableActivation["relu"] = 1] = "relu";
      FusableActivation[FusableActivation["relu6"] = 2] = "relu6";
      FusableActivation[FusableActivation["prelu"] = 3] = "prelu";
  })(FusableActivation || (FusableActivation = {}));

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmFusedMatMul;
  function setup(backend) {
      wasmFusedMatMul = backend.wasm.cwrap('_FusedMatMul', null /* void */, [
          'number',
          'array',
          'number',
          'number',
          'array',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number' // out_id
      ]);
  }
  function fusedBatchMatMul(args) {
      const { inputs, backend, attrs } = args;
      const { a, b, bias, preluActivationWeights } = inputs;
      if (a.dtype !== 'float32' || b.dtype !== 'float32') {
          throw new Error(`_FusedMatMul for non non-float32 tensors not yet supported.`);
      }
      const { transposeA, transposeB, activation } = attrs;
      const aId = backend.dataIdMap.get(a.dataId).id;
      const bId = backend.dataIdMap.get(b.dataId).id;
      let biasId = 0;
      if (bias != null) {
          const biasData = backend.dataIdMap.get(bias.dataId);
          if (biasData.shape.length !== 1) {
              throw new Error(`_FusedMatMul only supports rank-1 bias but got ` +
                  `rank ${biasData.shape.length}.`);
          }
          biasId = biasData.id;
      }
      const preluActivationWeightsId = preluActivationWeights == null ?
          0 :
          backend.dataIdMap.get(preluActivationWeights.dataId).id;
      const fusedActivation = FusableActivation[activation];
      if (fusedActivation == null) {
          throw new Error(`${activation} activation not yet supported for FusedConv2D ` +
              `in the wasm backend.`);
      }
      const leftDim = transposeA ? a.shape[2] : a.shape[1];
      const rightDim = transposeB ? b.shape[1] : b.shape[2];
      const batchDim = a.shape[0];
      const out = backend.makeOutput([batchDim, leftDim, rightDim], a.dtype);
      const outId = backend.dataIdMap.get(out.dataId).id;
      const aShapeBytes = new Uint8Array(new Int32Array(a.shape).buffer);
      const bShapeBytes = new Uint8Array(new Int32Array(b.shape).buffer);
      wasmFusedMatMul(aId, aShapeBytes, a.shape.length, bId, bShapeBytes, b.shape.length, transposeA, transposeB, fusedActivation, biasId, preluActivationWeightsId, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: '_FusedMatMul',
      backendName: 'wasm',
      setupFunc: setup,
      kernelFunc: fusedBatchMatMul
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  function registerUnaryKernel(kernelName) {
      let wasmFunc;
      function setupFunc(backend) {
          wasmFunc =
              backend.wasm.cwrap(kernelName, null /* void */, ['number', 'number']);
      }
      function kernelFunc(args) {
          const { backend, inputs: { x } } = args;
          const xId = backend.dataIdMap.get(x.dataId).id;
          const out = backend.makeOutput(x.shape, x.dtype);
          const outId = backend.dataIdMap.get(out.dataId).id;
          // Short-circuit zero-sized tensors.
          if (tfjsCore.util.sizeFromShape(out.shape) === 0) {
              return out;
          }
          wasmFunc(xId, outId);
          return out;
      }
      tfjsCore.registerKernel({ kernelName, backendName: 'wasm', setupFunc, kernelFunc });
  }

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  registerUnaryKernel('Abs');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  function registerBinaryKernel(kernelName, supportsFullBroadcast, dtype) {
      let wasmFunc;
      function setupFunc(backend) {
          wasmFunc = backend.wasm.cwrap(kernelName, null /* void */, [
              'number',
              'array',
              'number',
              'number',
              'array',
              'number',
              'number',
              'number' // out_id
          ]);
      }
      function kernelFunc(args) {
          const { backend, inputs } = args;
          const { a, b } = inputs;
          const aId = backend.dataIdMap.get(a.dataId).id;
          const bId = backend.dataIdMap.get(b.dataId).id;
          const outputType = dtype != null ? dtype : a.dtype;
          const newShape = tfjsCore.backend_util.assertAndGetBroadcastShape(a.shape, b.shape);
          const out = backend.makeOutput(newShape, outputType);
          // Short-circuit zero-sized tensors.
          if (tfjsCore.util.sizeFromShape(newShape) === 0) {
              return out;
          }
          const aShapeBytes = new Uint8Array(new Int32Array(a.shape).buffer);
          const bShapeBytes = new Uint8Array(new Int32Array(b.shape).buffer);
          const outId = backend.dataIdMap.get(out.dataId).id;
          const kernelFunc = () => wasmFunc(aId, aShapeBytes, a.shape.length, bId, bShapeBytes, b.shape.length, CppDType[a.dtype], outId);
          if (supportsFullBroadcast) {
              kernelFunc();
              return out;
          }
          const aBroadcastDims = tfjsCore.backend_util.getBroadcastDims(a.shape, newShape);
          const bBroadcastDims = tfjsCore.backend_util.getBroadcastDims(b.shape, newShape);
          const loopsOverAllOfA = aBroadcastDims.every((v, i) => v === i);
          const loopsOverAllOfB = bBroadcastDims.every((v, i) => v === i);
          if (loopsOverAllOfA && loopsOverAllOfB) {
              kernelFunc();
              return out;
          }
          else {
              throw new Error(`Broadcasting along outer dims is not yet ` +
                  `supported for ${kernelName}.`);
          }
      }
      tfjsCore.registerKernel({ kernelName, backendName: 'wasm', setupFunc, kernelFunc });
  }

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  const supportsFullBroadcast = true;
  registerBinaryKernel('Add', supportsFullBroadcast);

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmFunc;
  function setupFunc(backend) {
      wasmFunc = backend.wasm.cwrap('AddN', null /* void */, [
          'array',
          'number',
          'number',
          'number',
      ]);
  }
  function addn(args) {
      const { inputs, backend } = args;
      const out = backend.makeOutput(inputs[0].shape, inputs[0].dtype);
      // Short-circuit zero-sized tensors.
      if (tfjsCore.util.sizeFromShape(out.shape) === 0) {
          return out;
      }
      const inputIds = inputs.map(x => backend.dataIdMap.get(x.dataId).id);
      const inputIdsBytes = new Uint8Array(new Int32Array(inputIds).buffer);
      const outId = backend.dataIdMap.get(out.dataId).id;
      wasmFunc(inputIdsBytes, inputIds.length, CppDType[out.dtype], outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'AddN',
      backendName: 'wasm',
      setupFunc,
      kernelFunc: addn,
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmFunc$1;
  function setup$1(backend) {
      wasmFunc$1 = backend.wasm.cwrap('ArgMax', null /* void */, [
          'number',
          'number',
          'number',
          'number',
          'number' // out_id
      ]);
  }
  function argmax(args) {
      const { inputs: { x }, backend, attrs: { axis } } = args;
      const outShape = x.shape.slice(0, -1);
      const out = backend.makeOutput(outShape, 'int32');
      const xId = backend.dataIdMap.get(x.dataId).id;
      const outId = backend.dataIdMap.get(out.dataId).id;
      const outerSize = tfjsCore.util.sizeFromShape(out.shape);
      const innerSize = x.shape[axis];
      wasmFunc$1(xId, CppDType[x.dtype], outerSize, innerSize, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'ArgMax',
      backendName: 'wasm',
      kernelFunc: argmax,
      setupFunc: setup$1
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmAvgPool;
  function setup$2(backend) {
      wasmAvgPool = backend.wasm.cwrap('AvgPool', null /* void */, [
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
      ]);
  }
  function avgPool(args) {
      const { inputs, attrs, backend } = args;
      const convInfo = attrs;
      const { x } = inputs;
      const xId = backend.dataIdMap.get(x.dataId).id;
      const filterHeight = convInfo.filterHeight;
      const filterWidth = convInfo.filterWidth;
      const padTop = convInfo.padInfo.top;
      const padRight = convInfo.padInfo.right;
      const padBottom = convInfo.padInfo.bottom;
      const padLeft = convInfo.padInfo.left;
      const strideHeight = convInfo.strideHeight;
      const strideWidth = convInfo.strideWidth;
      const channels = convInfo.inChannels;
      if (convInfo.dataFormat !== 'channelsLast') {
          throw new Error(`wasm backend does not support dataFormat:'` +
              `${convInfo.dataFormat}'. Please use 'channelsLast'.`);
      }
      if (convInfo.dilationWidth !== 1 || convInfo.dilationHeight !== 1) {
          throw new Error(`was backend only supports average pooling with dilation = [1, 1], ` +
              `got [${convInfo.dilationHeight}, ${convInfo.dilationWidth}].`);
      }
      const out = backend.makeOutput(convInfo.outShape, 'float32');
      const outId = backend.dataIdMap.get(out.dataId).id;
      wasmAvgPool(xId, x.shape[0], x.shape[1], x.shape[2], filterHeight, filterWidth, padTop, padRight, padBottom, padLeft, strideHeight, strideWidth, channels, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'AvgPool',
      backendName: 'wasm',
      setupFunc: setup$2,
      kernelFunc: avgPool
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmBatchMatMul;
  function setup$3(backend) {
      wasmBatchMatMul = backend.wasm.cwrap('BatchMatMul', null /* void */, [
          'number',
          'array',
          'number',
          'number',
          'array',
          'number',
          'number',
          'number',
          'number' // out_id
      ]);
  }
  function batchMatMul(args) {
      const { inputs, backend, attrs } = args;
      const { a, b } = inputs;
      if (a.dtype !== 'float32' || b.dtype !== 'float32') {
          throw new Error(`BatchMatMul for non non-float32 tensors not yet supported.`);
      }
      const { transposeA, transposeB } = attrs;
      const aId = backend.dataIdMap.get(a.dataId).id;
      const bId = backend.dataIdMap.get(b.dataId).id;
      const leftDim = transposeA ? a.shape[2] : a.shape[1];
      const rightDim = transposeB ? b.shape[1] : b.shape[2];
      const batchDim = a.shape[0];
      const out = backend.makeOutput([batchDim, leftDim, rightDim], a.dtype);
      const outId = backend.dataIdMap.get(out.dataId).id;
      const aShapeBytes = new Uint8Array(new Int32Array(a.shape).buffer);
      const bShapeBytes = new Uint8Array(new Int32Array(b.shape).buffer);
      wasmBatchMatMul(aId, aShapeBytes, a.shape.length, bId, bShapeBytes, b.shape.length, transposeA, transposeB, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'BatchMatMul',
      backendName: 'wasm',
      setupFunc: setup$3,
      kernelFunc: batchMatMul
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  function cast(args) {
      const { inputs: { x }, attrs: { dtype }, backend } = args;
      const out = backend.makeOutput(x.shape, dtype);
      const inVals = backend.typedArrayFromHeap(x);
      const outVals = backend.typedArrayFromHeap(out);
      outVals.set(inVals);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'Cast',
      backendName: 'wasm',
      kernelFunc: cast,
  });

  /**
   * @license
   * Copyright 2019 Google LLC. All Rights Reserved.
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
  let wasmClip;
  function setup$4(backend) {
      wasmClip = backend.wasm.cwrap('ClipByValue', null /* void */, [
          'number',
          'number',
          'number',
          'number' // out_id
      ]);
  }
  function clip(args) {
      const { inputs, backend, attrs } = args;
      const { x } = inputs;
      const { min, max } = attrs;
      const xId = backend.dataIdMap.get(x.dataId).id;
      const out = backend.makeOutput(x.shape, 'float32');
      const outId = backend.dataIdMap.get(out.dataId).id;
      wasmClip(xId, min, max, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'ClipByValue',
      backendName: 'wasm',
      setupFunc: setup$4,
      kernelFunc: clip
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  function concat(args) {
      const { inputs, backend, attrs: { axis } } = args;
      const outShape = tfjsCore.backend_util.computeOutShape(inputs.map(t => t.shape), axis);
      const out = backend.makeOutput(outShape, inputs[0].dtype);
      const batchDim = tfjsCore.util.sizeFromShape(inputs[0].shape.slice(0, axis));
      let sumInnerDims = 0;
      const innerDims = inputs.map(input => {
          const innerDim = tfjsCore.util.sizeFromShape(input.shape.slice(axis));
          sumInnerDims += innerDim;
          return innerDim;
      });
      const inVals = inputs.map(input => backend.typedArrayFromHeap(input));
      const outVals = backend.typedArrayFromHeap(out);
      for (let b = 0; b < batchDim; b++) {
          let outOffset = b * sumInnerDims;
          for (let i = 0; i < inVals.length; i++) {
              const innerDim = innerDims[i];
              const inOffset = b * innerDim;
              const vals = inVals[i].subarray(inOffset, inOffset + innerDim);
              outVals.set(vals, outOffset);
              outOffset += innerDim;
          }
      }
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'Concat',
      backendName: 'wasm',
      kernelFunc: concat,
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmConv2d;
  function setup$5(backend) {
      wasmConv2d = backend.wasm.cwrap('Conv2D', null /* void */, [
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
      ]);
  }
  function conv2d(args) {
      const { inputs, attrs, backend } = args;
      const { x, filter } = inputs;
      const xId = backend.dataIdMap.get(x.dataId).id;
      const filterId = backend.dataIdMap.get(filter.dataId).id;
      const { strides, dilations, pad, dimRoundingMode, dataFormat } = attrs;
      const $dataFormat = tfjsCore.backend_util.convertConv2DDataFormat(dataFormat);
      const convInfo = tfjsCore.backend_util.computeConv2DInfo(x.shape, filter.shape, strides, dilations, pad, dimRoundingMode, false, $dataFormat);
      const filterHeight = convInfo.filterHeight;
      const filterWidth = convInfo.filterWidth;
      const padTop = convInfo.padInfo.top;
      const padRight = convInfo.padInfo.right;
      const padBottom = convInfo.padInfo.bottom;
      const padLeft = convInfo.padInfo.left;
      const dilationHeight = convInfo.dilationHeight;
      const dilationWidth = convInfo.dilationWidth;
      const strideHeight = convInfo.strideHeight;
      const strideWidth = convInfo.strideWidth;
      const inputChannels = convInfo.inChannels;
      const outputChannels = convInfo.outChannels;
      const isSamePad = convInfo.padInfo.type === 'SAME' ? 1 : 0;
      if (convInfo.dataFormat !== 'channelsLast') {
          throw new Error(`wasm backend Conv2D does not support dataFormat:'` +
              `${convInfo.dataFormat}'. Please use 'channelsLast'.`);
      }
      const out = backend.makeOutput(convInfo.outShape, 'float32');
      const outId = backend.dataIdMap.get(out.dataId).id;
      wasmConv2d(xId, x.shape[0], x.shape[1], x.shape[2], filterId, filterHeight, filterWidth, padTop, padRight, padBottom, padLeft, isSamePad, dilationHeight, dilationWidth, strideHeight, strideWidth, inputChannels, outputChannels, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'Conv2D',
      backendName: 'wasm',
      setupFunc: setup$5,
      kernelFunc: conv2d
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  registerUnaryKernel('Cos');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  // Must match enum in CropAndResize.cc
  var InterpolationMethod;
  (function (InterpolationMethod) {
      InterpolationMethod[InterpolationMethod["bilinear"] = 0] = "bilinear";
      InterpolationMethod[InterpolationMethod["nearest"] = 1] = "nearest";
  })(InterpolationMethod || (InterpolationMethod = {}));
  let wasmCropAndResize;
  function setup$6(backend) {
      wasmCropAndResize = backend.wasm.cwrap('CropAndResize', null /*void*/, [
          'number',
          'number',
          'number',
          'number',
          'array',
          'number',
          'number',
          'number',
          'number',
          'number' // out id
      ]);
  }
  function cropAndResize(args) {
      const { backend, inputs, attrs } = args;
      const { method, extrapolationValue, cropSize } = attrs;
      const { images, boxes, boxInd } = inputs;
      const numBoxes = boxes.shape[0];
      const [cropHeight, cropWidth] = cropSize;
      const outShape = [numBoxes, cropHeight, cropWidth, images.shape[3]];
      let imagesData = backend.dataIdMap.get(images.dataId);
      let castedData;
      if (images.dtype !== 'float32') {
          castedData =
              cast({ backend, inputs: { x: images }, attrs: { dtype: 'float32' } });
          imagesData = backend.dataIdMap.get(castedData.dataId);
      }
      const imagesId = imagesData.id;
      const boxesId = backend.dataIdMap.get(boxes.dataId).id;
      const boxIndId = backend.dataIdMap.get(boxInd.dataId).id;
      const out = backend.makeOutput(outShape, 'float32');
      const outId = backend.dataIdMap.get(out.dataId).id;
      const imagesShapeBytes = new Uint8Array(new Int32Array(images.shape).buffer);
      wasmCropAndResize(imagesId, boxesId, boxIndId, numBoxes, imagesShapeBytes, cropHeight, cropWidth, InterpolationMethod[method], extrapolationValue, outId);
      if (castedData != null) {
          backend.disposeData(castedData.dataId);
      }
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'CropAndResize',
      backendName: 'wasm',
      setupFunc: setup$6,
      kernelFunc: cropAndResize
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmDepthwiseConv2d;
  function setup$7(backend) {
      wasmDepthwiseConv2d =
          backend.wasm.cwrap('DepthwiseConv2dNative', null /* void */, [
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
          ]);
  }
  function depthwiseConv2d(args) {
      const { inputs, attrs, backend } = args;
      const { x, filter } = inputs;
      const xId = backend.dataIdMap.get(x.dataId).id;
      const filterId = backend.dataIdMap.get(filter.dataId).id;
      const { strides, dilations, pad, dimRoundingMode } = attrs;
      const $dilations = dilations == null ? [1, 1] : dilations;
      const convInfo = tfjsCore.backend_util.computeConv2DInfo(x.shape, filter.shape, strides, $dilations, pad, dimRoundingMode, true /* depthwise */);
      const filterHeight = convInfo.filterHeight;
      const filterWidth = convInfo.filterWidth;
      const padTop = convInfo.padInfo.top;
      const padRight = convInfo.padInfo.right;
      const padBottom = convInfo.padInfo.bottom;
      const padLeft = convInfo.padInfo.left;
      const dilationHeight = convInfo.dilationHeight;
      const dilationWidth = convInfo.dilationWidth;
      const strideHeight = convInfo.strideHeight;
      const strideWidth = convInfo.strideWidth;
      const inputChannels = convInfo.inChannels;
      const outputChannels = convInfo.outChannels;
      const isSamePad = convInfo.padInfo.type === 'SAME' ? 1 : 0;
      if (convInfo.dataFormat !== 'channelsLast') {
          throw new Error(`wasm backend DepthwiseConv2dNative does not support dataFormat:'` +
              `${convInfo.dataFormat}'. Please use 'channelsLast'.`);
      }
      const out = backend.makeOutput(convInfo.outShape, 'float32');
      const outId = backend.dataIdMap.get(out.dataId).id;
      wasmDepthwiseConv2d(xId, x.shape[0], x.shape[1], x.shape[2], filterId, filterHeight, filterWidth, padTop, padRight, padBottom, padLeft, isSamePad, dilationHeight, dilationWidth, strideHeight, strideWidth, inputChannels, outputChannels, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'DepthwiseConv2dNative',
      backendName: 'wasm',
      setupFunc: setup$7,
      kernelFunc: depthwiseConv2d
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  const supportsFullBroadcast$1 = false;
  registerBinaryKernel('Div', supportsFullBroadcast$1);

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  registerUnaryKernel('Exp');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  const supportsFullBroadcast$2 = false;
  registerBinaryKernel('FloorDiv', supportsFullBroadcast$2);

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmBatchNorm;
  function setup$8(backend) {
      wasmBatchNorm = backend.wasm.cwrap('FusedBatchNorm', null /* void */, ['number', 'number', 'number', 'number', 'number', 'number', 'number']);
  }
  function fusedBatchNorm(args) {
      const { backend, inputs, attrs } = args;
      const { varianceEpsilon } = attrs;
      const { x, mean, variance, offset, scale } = inputs;
      const xId = backend.dataIdMap.get(x.dataId).id;
      const meanId = backend.dataIdMap.get(mean.dataId).id;
      const varianceId = backend.dataIdMap.get(variance.dataId).id;
      const offsetId = offset != null ? backend.dataIdMap.get(offset.dataId).id : 0;
      const scaleId = scale != null ? backend.dataIdMap.get(scale.dataId).id : 0;
      const out = backend.makeOutput(x.shape, x.dtype);
      // Short-circuit zero-sized tensors.
      if (tfjsCore.util.sizeFromShape(x.shape) === 0) {
          return out;
      }
      const outId = backend.dataIdMap.get(out.dataId).id;
      wasmBatchNorm(xId, meanId, varianceId, offsetId, scaleId, varianceEpsilon, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'FusedBatchNorm',
      backendName: 'wasm',
      setupFunc: setup$8,
      kernelFunc: fusedBatchNorm
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmFusedConv2d;
  function setup$9(backend) {
      wasmFusedConv2d = backend.wasm.cwrap('FusedConv2D', null /* void */, [
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
      ]);
  }
  function fusedConv2d(args) {
      const { inputs, attrs, backend } = args;
      const { convInfo, activation } = attrs;
      const fusedActivation = FusableActivation[activation];
      if (fusedActivation == null) {
          throw new Error(`${activation} activation not yet supported for FusedConv2D ` +
              `in the wasm backend.`);
      }
      const { x, filter, bias, preluActivationWeights } = inputs;
      const xId = backend.dataIdMap.get(x.dataId).id;
      const filterId = backend.dataIdMap.get(filter.dataId).id;
      const outputChannels = convInfo.outChannels;
      let biasId = 0;
      if (bias != null) {
          const biasData = backend.dataIdMap.get(bias.dataId);
          if (biasData.shape.length !== 1) {
              throw new Error(`FusedConv2D only supports rank-1 bias but got ` +
                  `rank ${biasData.shape.length}.`);
          }
          if (biasData.shape[0] !== outputChannels) {
              throw new Error(`FusedConv2D bias shape (${biasData.shape}) does not ` +
                  `match the number of output channels (${outputChannels})`);
          }
          biasId = biasData.id;
      }
      const filterHeight = convInfo.filterHeight;
      const filterWidth = convInfo.filterWidth;
      const padTop = convInfo.padInfo.top;
      const padRight = convInfo.padInfo.right;
      const padBottom = convInfo.padInfo.bottom;
      const padLeft = convInfo.padInfo.left;
      const dilationHeight = convInfo.dilationHeight;
      const dilationWidth = convInfo.dilationWidth;
      const strideHeight = convInfo.strideHeight;
      const strideWidth = convInfo.strideWidth;
      const inputChannels = convInfo.inChannels;
      const isSamePad = convInfo.padInfo.type === 'SAME' ? 1 : 0;
      const batchSize = convInfo.batchSize;
      const inHeight = convInfo.inHeight;
      const inWidth = convInfo.inWidth;
      if (convInfo.dataFormat !== 'channelsLast') {
          throw new Error(`wasm backend FusedConv2D does not support dataFormat:'` +
              `${convInfo.dataFormat}'. Please use 'channelsLast'.`);
      }
      const out = backend.makeOutput(convInfo.outShape, 'float32');
      const outId = backend.dataIdMap.get(out.dataId).id;
      const preluActivationWeightsId = preluActivationWeights == null ?
          0 :
          backend.dataIdMap.get(preluActivationWeights.dataId).id;
      wasmFusedConv2d(xId, batchSize, inHeight, inWidth, filterId, filterHeight, filterWidth, biasId, padTop, padRight, padBottom, padLeft, isSamePad, dilationHeight, dilationWidth, strideHeight, strideWidth, inputChannels, outputChannels, fusedActivation, preluActivationWeightsId, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'FusedConv2D',
      backendName: 'wasm',
      setupFunc: setup$9,
      kernelFunc: fusedConv2d
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmFusedDepthwiseConv2d;
  function setup$a(backend) {
      wasmFusedDepthwiseConv2d =
          backend.wasm.cwrap('FusedDepthwiseConv2D', null /* void */, [
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
              'number',
          ]);
  }
  function fusedDepthwiseConv2d(args) {
      const { inputs, attrs, backend } = args;
      const { convInfo, activation } = attrs;
      const fusedActivation = FusableActivation[activation];
      if (fusedActivation == null) {
          throw new Error(`${activation} activation not yet supported for FusedDepthwiseConv2D ` +
              `in the wasm backend.`);
      }
      const { x, filter, bias, preluActivationWeights } = inputs;
      const xId = backend.dataIdMap.get(x.dataId).id;
      const filterId = backend.dataIdMap.get(filter.dataId).id;
      const outputChannels = convInfo.outChannels;
      let biasId = 0;
      if (bias != null) {
          const biasData = backend.dataIdMap.get(bias.dataId);
          if (biasData.shape.length !== 1) {
              throw new Error(`FusedDepthwiseConv2D only supports rank-1 bias but got ` +
                  `rank ${biasData.shape.length}.`);
          }
          if (biasData.shape[0] !== outputChannels) {
              throw new Error(`FusedDepthwiseConv2D bias shape (${biasData.shape}) does not ` +
                  `match the number of output channels (${outputChannels})`);
          }
          biasId = biasData.id;
      }
      const filterHeight = convInfo.filterHeight;
      const filterWidth = convInfo.filterWidth;
      const padTop = convInfo.padInfo.top;
      const padRight = convInfo.padInfo.right;
      const padBottom = convInfo.padInfo.bottom;
      const padLeft = convInfo.padInfo.left;
      const dilationHeight = convInfo.dilationHeight;
      const dilationWidth = convInfo.dilationWidth;
      const strideHeight = convInfo.strideHeight;
      const strideWidth = convInfo.strideWidth;
      const inputChannels = convInfo.inChannels;
      const isSamePad = convInfo.padInfo.type === 'SAME' ? 1 : 0;
      const batchSize = convInfo.batchSize;
      const inHeight = convInfo.inHeight;
      const inWidth = convInfo.inWidth;
      if (convInfo.dataFormat !== 'channelsLast') {
          throw new Error(`wasm backend FusedDepthwiseConv2D does not support dataFormat:'` +
              `${convInfo.dataFormat}'. Please use 'channelsLast'.`);
      }
      const out = backend.makeOutput(convInfo.outShape, 'float32');
      const outId = backend.dataIdMap.get(out.dataId).id;
      const preluActivationWeightsId = preluActivationWeights == null ?
          0 :
          backend.dataIdMap.get(preluActivationWeights.dataId).id;
      wasmFusedDepthwiseConv2d(xId, batchSize, inHeight, inWidth, filterId, filterHeight, filterWidth, biasId, padTop, padRight, padBottom, padLeft, isSamePad, dilationHeight, dilationWidth, strideHeight, strideWidth, inputChannels, outputChannels, fusedActivation, preluActivationWeightsId, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'FusedDepthwiseConv2D',
      backendName: 'wasm',
      setupFunc: setup$a,
      kernelFunc: fusedDepthwiseConv2d
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmGather;
  function setup$b(backend) {
      wasmGather = backend.wasm.cwrap('Gather', null /*void*/, [
          'number',
          'number',
          'array',
          'number',
          'number',
          'number',
          'array',
          'number' // outId
      ]);
  }
  function gather(args) {
      const { backend, inputs, attrs } = args;
      const { x, indices } = inputs;
      const { axis } = attrs;
      const newShape = x.shape.slice();
      newShape[axis] = tfjsCore.util.sizeFromShape(indices.shape);
      const stridesSize = x.shape.length - 1;
      const out = backend.makeOutput(newShape, x.dtype);
      if (tfjsCore.util.sizeFromShape(x.shape) === 0) {
          return out;
      }
      const xData = backend.dataIdMap.get(x.dataId);
      const xId = xData.id;
      const indicesData = backend.dataIdMap.get(indices.dataId);
      const indicesId = indicesData.id;
      const outId = backend.dataIdMap.get(out.dataId).id;
      const xStridesBytes = new Uint8Array(new Int32Array(tfjsCore.util.computeStrides(x.shape)).buffer);
      const outStridesBytes = new Uint8Array(new Int32Array(tfjsCore.util.computeStrides(newShape)).buffer);
      wasmGather(xId, CppDType[x.dtype], xStridesBytes, stridesSize, indicesId, axis, outStridesBytes, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'Gather',
      backendName: 'wasm',
      setupFunc: setup$b,
      kernelFunc: gather
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmGatherNd;
  function setup$c(backend) {
      wasmGatherNd = backend.wasm.cwrap('GatherNd', null /*void*/, [
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'array',
          'number' // outId
      ]);
  }
  function gatherNd(args) {
      const { backend, inputs } = args;
      const { x, indices } = inputs;
      const [resultShape, numSlices, sliceSize, strides] = tfjsCore.gather_util.prepareAndValidate(x, indices);
      const out = backend.makeOutput(resultShape, x.dtype);
      if (numSlices === 0) {
          return out;
      }
      const indicesShape = indices.shape;
      const sliceRank = indicesShape[indicesShape.length - 1];
      const xData = backend.dataIdMap.get(x.dataId);
      const xId = xData.id;
      const indicesData = backend.dataIdMap.get(indices.dataId);
      const indicesId = indicesData.id;
      const stridesBytes = new Uint8Array(new Int32Array(strides).buffer);
      const outId = backend.dataIdMap.get(out.dataId).id;
      wasmGatherNd(xId, CppDType[x.dtype], indicesId, numSlices, sliceRank, sliceSize, stridesBytes, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'GatherNd',
      backendName: 'wasm',
      setupFunc: setup$c,
      kernelFunc: gatherNd
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  const supportsFullBroadcast$3 = false;
  registerBinaryKernel('Greater', supportsFullBroadcast$3, 'bool');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  const supportsFullBroadcast$4 = false;
  registerBinaryKernel('GreaterEqual', supportsFullBroadcast$4, 'bool');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  const supportsFullBroadcast$5 = false;
  registerBinaryKernel('Less', supportsFullBroadcast$5, 'bool');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  const supportsFullBroadcast$6 = false;
  registerBinaryKernel('LessEqual', supportsFullBroadcast$6, 'bool');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  registerUnaryKernel('Log');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  const supportsFullBroadcast$7 = false;
  registerBinaryKernel('LogicalAnd', supportsFullBroadcast$7, 'bool');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmMax;
  function setup$d(backend) {
      wasmMax =
          backend.wasm.cwrap('Max', null /*void*/, ['number, number, number']);
  }
  function max(args) {
      const { backend, inputs, attrs } = args;
      const { reductionIndices } = attrs;
      const { x } = inputs;
      const xId = backend.dataIdMap.get(x.dataId).id;
      const origAxes = tfjsCore.util.parseAxisParam(reductionIndices, x.shape);
      tfjsCore.backend_util.assertAxesAreInnerMostDims('max', origAxes, x.shape.length);
      const [outShape, reduceShape] = tfjsCore.backend_util.computeOutAndReduceShapes(x.shape, origAxes);
      const reduceSize = tfjsCore.util.sizeFromShape(reduceShape);
      const out = backend.makeOutput(outShape, x.dtype);
      if (tfjsCore.util.sizeFromShape(x.shape) === 0) {
          return out;
      }
      const outId = backend.dataIdMap.get(out.dataId).id;
      wasmMax(xId, reduceSize, outId);
      return out;
  }
  tfjsCore.registerKernel({ kernelName: tfjsCore.Max, backendName: 'wasm', setupFunc: setup$d, kernelFunc: max });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  const supportsFullBroadcast$8 = false;
  registerBinaryKernel('Maximum', supportsFullBroadcast$8);

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmMaxPool;
  function setup$e(backend) {
      wasmMaxPool = backend.wasm.cwrap('MaxPool', null /* void */, [
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
      ]);
  }
  function maxPool(args) {
      const { inputs, attrs, backend } = args;
      const convInfo = attrs;
      const { x } = inputs;
      const xId = backend.dataIdMap.get(x.dataId).id;
      const filterHeight = convInfo.filterHeight;
      const filterWidth = convInfo.filterWidth;
      const padTop = convInfo.padInfo.top;
      const padRight = convInfo.padInfo.right;
      const padBottom = convInfo.padInfo.bottom;
      const padLeft = convInfo.padInfo.left;
      const dilationHeight = convInfo.dilationHeight;
      const dilationWidth = convInfo.dilationWidth;
      const strideHeight = convInfo.strideHeight;
      const strideWidth = convInfo.strideWidth;
      const inputChannels = convInfo.inChannels;
      const outputChannels = convInfo.outChannels;
      if (convInfo.dataFormat !== 'channelsLast') {
          throw new Error(`wasm backend does not support dataFormat:'` +
              `${convInfo.dataFormat}'. Please use 'channelsLast'.`);
      }
      const out = backend.makeOutput(convInfo.outShape, 'float32');
      const outId = backend.dataIdMap.get(out.dataId).id;
      wasmMaxPool(xId, x.shape[0], x.shape[1], x.shape[2], filterHeight, filterWidth, padTop, padRight, padBottom, padLeft, dilationHeight, dilationWidth, strideHeight, strideWidth, inputChannels, outputChannels, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'MaxPool',
      backendName: 'wasm',
      setupFunc: setup$e,
      kernelFunc: maxPool
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmMin;
  function setup$f(backend) {
      wasmMin =
          backend.wasm.cwrap('Min', null /*void*/, ['number, number, number']);
  }
  function min(args) {
      const { backend, inputs, attrs } = args;
      const { axes } = attrs;
      const { x } = inputs;
      const xId = backend.dataIdMap.get(x.dataId).id;
      tfjsCore.backend_util.assertAxesAreInnerMostDims('min', axes, x.shape.length);
      const [outShape, reduceShape] = tfjsCore.backend_util.computeOutAndReduceShapes(x.shape, axes);
      const reduceSize = tfjsCore.util.sizeFromShape(reduceShape);
      const out = backend.makeOutput(outShape, x.dtype);
      if (tfjsCore.util.sizeFromShape(x.shape) === 0) {
          return out;
      }
      const outId = backend.dataIdMap.get(out.dataId).id;
      wasmMin(xId, reduceSize, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'Min',
      backendName: 'wasm',
      setupFunc: setup$f,
      kernelFunc: min
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  const supportsFullBroadcast$9 = false;
  registerBinaryKernel('Minimum', supportsFullBroadcast$9);

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  const supportsFullBroadcast$a = true;
  registerBinaryKernel('Mul', supportsFullBroadcast$a);

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
  registerUnaryKernel('Neg');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  /**
   * Parse the result of the c++ method, which has the shape equivalent to
   * `Result`.
   */
  function parseResultStruct(backend, resOffset) {
      const result = new Int32Array(backend.wasm.HEAPU8.buffer, resOffset, 3);
      const pSelectedIndices = result[0];
      const selectedSize = result[1];
      const pSelectedScores = result[2];
      // Since the result was allocated on the heap, we have to delete it.
      backend.wasm._free(resOffset);
      return { pSelectedIndices, selectedSize, pSelectedScores };
  }

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmFunc$2;
  function setup$g(backend) {
      wasmFunc$2 = backend.wasm.cwrap('NonMaxSuppressionV3', 'number', // Result*
      [
          'number',
          'number',
          'number',
          'number',
          'number',
      ]);
  }
  function kernelFunc(args) {
      const { backend, inputs, attrs } = args;
      const { iouThreshold, maxOutputSize, scoreThreshold } = attrs;
      const { boxes, scores } = inputs;
      const boxesId = backend.dataIdMap.get(boxes.dataId).id;
      const scoresId = backend.dataIdMap.get(scores.dataId).id;
      const resOffset = wasmFunc$2(boxesId, scoresId, maxOutputSize, iouThreshold, scoreThreshold);
      const { pSelectedIndices, selectedSize, pSelectedScores } = parseResultStruct(backend, resOffset);
      // Since we are not using scores for V3, we have to delete it from the heap.
      backend.wasm._free(pSelectedScores);
      const selectedIndicesTensor = backend.makeOutput([selectedSize], 'int32', pSelectedIndices);
      return selectedIndicesTensor;
  }
  tfjsCore.registerKernel({
      kernelName: 'NonMaxSuppressionV3',
      backendName: 'wasm',
      setupFunc: setup$g,
      kernelFunc,
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmFunc$3;
  function setup$h(backend) {
      wasmFunc$3 = backend.wasm.cwrap('NonMaxSuppressionV5', 'number', // Result*
      [
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
      ]);
  }
  function kernelFunc$1(args) {
      const { backend, inputs, attrs } = args;
      const { iouThreshold, maxOutputSize, scoreThreshold, softNmsSigma } = attrs;
      const { boxes, scores } = inputs;
      const boxesId = backend.dataIdMap.get(boxes.dataId).id;
      const scoresId = backend.dataIdMap.get(scores.dataId).id;
      const resOffset = wasmFunc$3(boxesId, scoresId, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma);
      const { pSelectedIndices, selectedSize, pSelectedScores, } = parseResultStruct(backend, resOffset);
      const selectedIndicesTensor = backend.makeOutput([selectedSize], 'int32', pSelectedIndices);
      const selectedScoresTensor = backend.makeOutput([selectedSize], 'float32', pSelectedScores);
      return [selectedIndicesTensor, selectedScoresTensor];
  }
  tfjsCore.registerKernel({
      kernelName: 'NonMaxSuppressionV5',
      backendName: 'wasm',
      setupFunc: setup$h,
      kernelFunc: kernelFunc$1,
  });

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
  const supportsFullBroadcast$b = false;
  registerBinaryKernel('NotEqual', supportsFullBroadcast$b, 'bool');

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
  function onesLike(args) {
      const { inputs: { x }, backend } = args;
      const out = backend.makeOutput(x.shape, x.dtype);
      const outVals = backend.typedArrayFromHeap(out);
      outVals.fill(1);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'OnesLike',
      backendName: 'wasm',
      kernelFunc: onesLike,
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmPadV2;
  function setup$i(backend) {
      wasmPadV2 = backend.wasm.cwrap('PadV2', null /* void */, [
          'number',
          'array',
          'number',
          'number',
          'array',
          'number',
          'number',
      ]);
  }
  function pad(args) {
      const { inputs: { x }, backend, attrs: { paddings, constantValue } } = args;
      const outShape = paddings.map((p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);
      const xId = backend.dataIdMap.get(x.dataId).id;
      const out = backend.makeOutput(outShape, x.dtype);
      const outId = backend.dataIdMap.get(out.dataId).id;
      const xShapeBytes = new Uint8Array(new Int32Array(x.shape).buffer);
      const paddingsFlat = [].concat(...paddings);
      const paddingsBytes = new Uint8Array(new Int32Array(paddingsFlat).buffer);
      wasmPadV2(xId, xShapeBytes, x.shape.length, CppDType[x.dtype], paddingsBytes, constantValue, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'PadV2',
      backendName: 'wasm',
      kernelFunc: pad,
      setupFunc: setup$i
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  const supportsFullBroadcast$c = false;
  registerBinaryKernel('Pow', supportsFullBroadcast$c);

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmPrelu;
  function setup$j(backend) {
      wasmPrelu = backend.wasm.cwrap('Prelu', null /* void */, [
          'number',
          'number',
          'number' // out_id
      ]);
  }
  function prelu(args) {
      const { inputs, backend } = args;
      const { x, alpha } = inputs;
      const xId = backend.dataIdMap.get(x.dataId).id;
      const weightsId = backend.dataIdMap.get(alpha.dataId).id;
      const out = backend.makeOutput(x.shape, 'float32');
      const outId = backend.dataIdMap.get(out.dataId).id;
      wasmPrelu(xId, weightsId, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'Prelu',
      backendName: 'wasm',
      setupFunc: setup$j,
      kernelFunc: prelu
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  registerUnaryKernel('Relu');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  registerUnaryKernel('Relu6');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  function reshape(args) {
      const { inputs: { x }, attrs: { shape } } = args;
      return { dataId: x.dataId, shape, dtype: x.dtype };
  }
  tfjsCore.registerKernel({
      kernelName: 'Reshape',
      backendName: 'wasm',
      kernelFunc: reshape,
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmResizeBilinear;
  function setup$k(backend) {
      wasmResizeBilinear = backend.wasm.cwrap('ResizeBilinear', null /*void*/, [
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'number' // outId
      ]);
  }
  function resizeBilinear(args) {
      const { backend, inputs, attrs } = args;
      const { x } = inputs;
      const { alignCorners, newHeight, newWidth } = attrs;
      const [batch, oldHeight, oldWidth, numChannels] = x.shape;
      const outShape = [batch, newHeight, newWidth, numChannels];
      let xData = backend.dataIdMap.get(x.dataId);
      let castedData;
      if (xData.dtype !== 'float32') {
          castedData = cast({ backend, inputs: { x }, attrs: { dtype: 'float32' } });
          xData = backend.dataIdMap.get(castedData.dataId);
      }
      const xId = xData.id;
      const out = backend.makeOutput(outShape, 'float32');
      if (tfjsCore.util.sizeFromShape(x.shape) === 0) {
          return out;
      }
      const outId = backend.dataIdMap.get(out.dataId).id;
      wasmResizeBilinear(xId, batch, oldHeight, oldWidth, numChannels, newHeight, newWidth, alignCorners ? 1 : 0, outId);
      if (castedData != null) {
          backend.disposeData(castedData.dataId);
      }
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'ResizeBilinear',
      backendName: 'wasm',
      setupFunc: setup$k,
      kernelFunc: resizeBilinear
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  registerUnaryKernel('Rsqrt');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmScatterNd;
  function setup$l(backend) {
      wasmScatterNd = backend.wasm.cwrap('ScatterNd', null /*void*/, [
          'number',
          'number',
          'number',
          'number',
          'number',
          'number',
          'array',
          'number',
          'number' // outId
      ]);
  }
  function scatterNd(args) {
      const { backend, inputs, attrs } = args;
      const { indices, updates } = inputs;
      const { shape } = attrs;
      const out = backend.makeOutput(shape, updates.dtype);
      if (tfjsCore.util.sizeFromShape(shape) === 0) {
          return out;
      }
      const { sliceRank, numUpdates, sliceSize, strides, outputSize } = tfjsCore.scatter_util.calculateShapes(updates, indices, shape);
      const indicesData = backend.dataIdMap.get(indices.dataId);
      const indicesId = indicesData.id;
      const updatesData = backend.dataIdMap.get(updates.dataId);
      const updatesId = updatesData.id;
      const stridesBytes = new Uint8Array(new Int32Array(strides).buffer);
      const outId = backend.dataIdMap.get(out.dataId).id;
      wasmScatterNd(indicesId, updatesId, CppDType[updates.dtype], sliceRank, numUpdates, sliceSize, stridesBytes, outputSize, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'ScatterNd',
      backendName: 'wasm',
      setupFunc: setup$l,
      kernelFunc: scatterNd
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmFunc$4;
  function setup$m(backend) {
      wasmFunc$4 =
          backend.wasm.cwrap('Sigmoid', null /* void */, ['number', 'number']);
  }
  function sigmoid(args) {
      const { backend, inputs: { x } } = args;
      const xId = backend.dataIdMap.get(x.dataId).id;
      const out = backend.makeOutput(x.shape, x.dtype);
      const outId = backend.dataIdMap.get(out.dataId).id;
      // Short-circuit zero-sized tensors.
      if (tfjsCore.util.sizeFromShape(out.shape) === 0) {
          return out;
      }
      wasmFunc$4(xId, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'Sigmoid',
      backendName: 'wasm',
      setupFunc: setup$m,
      kernelFunc: sigmoid
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  registerUnaryKernel('Sin');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  function slice(args) {
      const { inputs: { x }, attrs: { begin, size }, backend } = args;
      const isContinous = tfjsCore.slice_util.isSliceContinous(x.shape, begin, size);
      const xVals = backend.typedArrayFromHeap(x);
      const out = backend.makeOutput(size, x.dtype);
      const outVals = backend.typedArrayFromHeap(out);
      const xStrides = tfjsCore.util.computeStrides(x.shape);
      if (isContinous) {
          const flatOffset = tfjsCore.slice_util.computeFlatOffset(begin, xStrides);
          outVals.set(xVals.subarray(flatOffset, flatOffset + tfjsCore.util.sizeFromShape(size)));
          return out;
      }
      const rank = x.shape.length;
      if (rank === 2) {
          slice2d(xVals, xStrides[0], outVals, begin, size);
      }
      else if (rank === 3) {
          slice3d(xVals, xStrides[0], xStrides[1], outVals, begin, size);
      }
      else if (rank === 4) {
          slice4d(xVals, xStrides[0], xStrides[1], xStrides[2], outVals, begin, size);
      }
      else {
          genericSliceSlow(xVals, x, outVals, begin, size);
      }
      return out;
  }
  function slice2d(xVals, xStride, outVals, begin, size) {
      let outOffset = 0;
      const beginI = begin[0];
      const beginJ = begin[1];
      const endI = beginI + size[0];
      for (let i = beginI; i < endI; i++) {
          const xOffset = i * xStride + beginJ;
          outVals.set(xVals.subarray(xOffset, xOffset + size[1]), outOffset);
          outOffset += size[1];
      }
  }
  function slice3d(xVals, xStride1, xStride2, outVals, begin, size) {
      let outOffset = 0;
      const beginI = begin[0];
      const beginJ = begin[1];
      const beginK = begin[2];
      const endI = beginI + size[0];
      const endJ = beginJ + size[1];
      for (let i = beginI; i < endI; i++) {
          for (let j = beginJ; j < endJ; j++) {
              const xOffset = i * xStride1 + j * xStride2 + beginK;
              outVals.set(xVals.subarray(xOffset, xOffset + size[2]), outOffset);
              outOffset += size[2];
          }
      }
  }
  function slice4d(xVals, xStride1, xStride2, xStride3, outVals, begin, size) {
      let outOffset = 0;
      const beginI = begin[0];
      const beginJ = begin[1];
      const beginK = begin[2];
      const endI = beginI + size[0];
      const endJ = beginJ + size[1];
      const endK = beginK + size[2];
      const beginL = begin[3];
      for (let i = beginI; i < endI; i++) {
          for (let j = beginJ; j < endJ; j++) {
              for (let k = beginK; k < endK; k++) {
                  const xOffset = i * xStride1 + j * xStride2 + k * xStride3 + beginL;
                  outVals.set(xVals.subarray(xOffset, xOffset + size[3]), outOffset);
                  outOffset += size[3];
              }
          }
      }
  }
  function genericSliceSlow(xVals, xInfo, outVals, begin, size) {
      const outBuf = tfjsCore.buffer(size, xInfo.dtype, outVals);
      const xBuf = tfjsCore.buffer(xInfo.shape, xInfo.dtype, xVals);
      for (let i = 0; i < outBuf.size; ++i) {
          const loc = outBuf.indexToLoc(i);
          const xLoc = loc.map((idx, j) => idx + begin[j]);
          outVals[i] = xBuf.get(...xLoc);
      }
  }
  tfjsCore.registerKernel({
      kernelName: 'Slice',
      backendName: 'wasm',
      kernelFunc: slice,
  });

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
  let wasmFunc$5;
  function setup$n(backend) {
      wasmFunc$5 = backend.wasm.cwrap('Softmax', null /* void */, [
          'number',
          'number',
          'number',
          'number' // batch
      ]);
  }
  function softmax(args) {
      const { backend, inputs: { logits }, attrs: { dim } } = args;
      const xId = backend.dataIdMap.get(logits.dataId).id;
      const out = backend.makeOutput(logits.shape, logits.dtype);
      const outId = backend.dataIdMap.get(out.dataId).id;
      const channels = logits.shape[dim];
      const batch = tfjsCore.util.sizeFromShape(logits.shape) / channels;
      // Short-circuit zero-sized tensors.
      if (tfjsCore.util.sizeFromShape(out.shape) === 0) {
          return out;
      }
      wasmFunc$5(xId, outId, channels, batch);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'Softmax',
      backendName: 'wasm',
      setupFunc: setup$n,
      kernelFunc: softmax
  });

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
  function split(args) {
      const { inputs, attrs, backend } = args;
      const { x } = inputs;
      const { numOrSizeSplits, axis } = attrs;
      const $axis = tfjsCore.util.parseAxisParam(axis, x.shape)[0];
      let splitSizes;
      if (typeof (numOrSizeSplits) === 'number') {
          splitSizes =
              new Array(numOrSizeSplits).fill(x.shape[$axis] / numOrSizeSplits);
      }
      else {
          splitSizes = numOrSizeSplits;
      }
      const begin = new Array(x.shape.length).fill(0);
      const size = x.shape.slice();
      return splitSizes.map(s => {
          const xSliceSize = [...size];
          xSliceSize[$axis] = s;
          const xSlice = slice({ inputs: { x }, attrs: { begin, size: xSliceSize }, backend });
          begin[$axis] += s;
          return xSlice;
      });
  }
  tfjsCore.registerKernel({ kernelName: tfjsCore.SplitV, backendName: 'wasm', kernelFunc: split });

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
  registerUnaryKernel('Sqrt');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  registerUnaryKernel('Square');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  const supportsFullBroadcast$d = true;
  registerBinaryKernel('Sub', supportsFullBroadcast$d);

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmSum;
  function setup$o(backend) {
      wasmSum =
          backend.wasm.cwrap('Sum', null /*void*/, ['number, number, number']);
  }
  function sum(args) {
      const { backend, inputs, attrs } = args;
      const { axes } = attrs;
      const { x } = inputs;
      const xId = backend.dataIdMap.get(x.dataId).id;
      tfjsCore.backend_util.assertAxesAreInnerMostDims('sum', axes, x.shape.length);
      const [outShape, reduceShape] = tfjsCore.backend_util.computeOutAndReduceShapes(x.shape, axes);
      const reduceSize = tfjsCore.util.sizeFromShape(reduceShape);
      const out = backend.makeOutput(outShape, x.dtype);
      if (tfjsCore.util.sizeFromShape(x.shape) === 0) {
          return out;
      }
      const outId = backend.dataIdMap.get(out.dataId).id;
      wasmSum(xId, reduceSize, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'Sum',
      backendName: 'wasm',
      setupFunc: setup$o,
      kernelFunc: sum
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  registerUnaryKernel('Tanh');

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmTile;
  function setup$p(backend) {
      wasmTile = backend.wasm.cwrap('Tile', null /* void */, [
          'number',
          'array',
          'number',
          'array',
          'number',
          'number' // out_id
      ]);
  }
  function tile(args) {
      const { inputs, backend, attrs } = args;
      const { x } = inputs;
      const xId = backend.dataIdMap.get(x.dataId).id;
      const { reps } = attrs;
      const newShape = new Array(x.shape.length);
      for (let i = 0; i < newShape.length; i++) {
          newShape[i] = x.shape[i] * reps[i];
      }
      const xShapeBytes = new Uint8Array(new Int32Array(x.shape).buffer);
      const newShapeBytes = new Uint8Array(new Int32Array(newShape).buffer);
      const out = backend.makeOutput(newShape, x.dtype);
      const outId = backend.dataIdMap.get(out.dataId).id;
      wasmTile(xId, xShapeBytes, x.shape.length, newShapeBytes, newShape.length, CppDType[out.dtype], outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'Tile',
      backendName: 'wasm',
      setupFunc: setup$p,
      kernelFunc: tile
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  let wasmTranspose;
  function setup$q(backend) {
      wasmTranspose = backend.wasm.cwrap('Transpose', null /* void */, [
          'number',
          'array',
          'number',
          'number',
          'number',
          'array',
          'number',
      ]);
  }
  function transpose(args) {
      const { inputs, backend, attrs } = args;
      // Reduce any dimensions with size one. Lower-rank transpose kernel performs
      // better due to simpler memory access pattern.
      const [reducedShape, perm] = removeOneSizeDims(inputs.x.shape, attrs.perm);
      const x = {
          dataId: inputs.x.dataId,
          shape: reducedShape,
          dtype: inputs.x.dtype
      };
      let permIsNoOp = true;
      for (let i = 0; i < perm.length; i++) {
          if (perm[i] !== i) {
              permIsNoOp = false;
          }
      }
      const outShape = computeOutShape(inputs.x.shape, attrs.perm);
      if (permIsNoOp) {
          return { dataId: x.dataId, shape: outShape, dtype: x.dtype };
      }
      const out = backend.makeOutput(outShape, x.dtype);
      const xId = backend.dataIdMap.get(x.dataId).id;
      const outId = backend.dataIdMap.get(out.dataId).id;
      const permBytes = new Uint8Array(new Int32Array(perm).buffer);
      const xShapeBytes = new Uint8Array(new Int32Array(x.shape).buffer);
      wasmTranspose(xId, xShapeBytes, x.shape.length, CppDType[x.dtype], outId, permBytes, perm.length);
      return out;
  }
  function computeOutShape(inShape, perm) {
      const outShape = new Array(inShape.length);
      for (let i = 0; i < outShape.length; i++) {
          outShape[i] = inShape[perm[i]];
      }
      return outShape;
  }
  function removeOneSizeDims(shape, perm) {
      const newShape = [];
      const newPerm = [];
      for (let i = 0; i < shape.length; ++i) {
          if (shape[i] !== 1) {
              newShape.push(shape[i]);
          }
          if (shape[perm[i]] !== 1) {
              newPerm.push(perm[i]);
          }
      }
      for (let i = 0; i < newPerm.length; ++i) {
          let minValIdx = -1;
          for (let j = 0; j < newPerm.length; ++j) {
              if (newPerm[j] >= i &&
                  (minValIdx === -1 || newPerm[minValIdx] > newPerm[j])) {
                  minValIdx = j;
              }
          }
          newPerm[minValIdx] = i;
      }
      return [newShape, newPerm];
  }
  tfjsCore.registerKernel({
      kernelName: 'Transpose',
      backendName: 'wasm',
      kernelFunc: transpose,
      setupFunc: setup$q,
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  function unpack(args) {
      const { inputs: { x }, backend, attrs: { axis } } = args;
      const numOutputs = x.shape[axis];
      const rank = x.shape.length;
      const outShape = new Array(rank - 1);
      let outIndex = 0;
      for (let i = 0; i < rank; i++) {
          if (i !== axis) {
              outShape[outIndex++] = x.shape[i];
          }
      }
      const outs = new Array(numOutputs);
      const begin = new Array(rank).fill(0);
      const size = x.shape.slice();
      size[axis] = 1;
      for (let i = 0; i < outs.length; i++) {
          begin[axis] = i;
          outs[i] = slice({ inputs: { x }, attrs: { begin, size }, backend });
      }
      return outs.map(({ dataId, dtype }) => ({ dataId, dtype, shape: outShape }));
  }
  tfjsCore.registerKernel({
      kernelName: 'Unpack',
      backendName: 'wasm',
      kernelFunc: unpack,
  });

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
  function zerosLike(args) {
      const { inputs: { x }, backend } = args;
      const out = backend.makeOutput(x.shape, x.dtype);
      const outVals = backend.typedArrayFromHeap(out);
      outVals.fill(0);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'ZerosLike',
      backendName: 'wasm',
      kernelFunc: zerosLike,
  });

  /**
   * @license
   * Copyright 2019 Google Inc. All Rights Reserved.
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
  const WASM_PRIORITY = 2;
  class BackendWasm extends tfjsCore.KernelBackend {
      constructor(wasm) {
          super();
          this.wasm = wasm;
          // 0 is reserved for null data ids.
          this.dataIdNextNumber = 1;
          this.wasm.tfjs.init();
          this.dataIdMap = new tfjsCore.DataStorage(this, tfjsCore.engine());
      }
      write(values, shape, dtype) {
          const dataId = {};
          this.move(dataId, values, shape, dtype);
          return dataId;
      }
      numDataIds() {
          return this.dataIdMap.numDataIds();
      }
      async time(f) {
          const start = tfjsCore.util.now();
          f();
          const kernelMs = tfjsCore.util.now() - start;
          return { kernelMs };
      }
      move(dataId, values, shape, dtype) {
          const id = this.dataIdNextNumber++;
          if (dtype === 'string') {
              const stringBytes = values;
              this.dataIdMap.set(dataId, { id, stringBytes, shape, dtype, memoryOffset: null });
              return;
          }
          const size = tfjsCore.util.sizeFromShape(shape);
          const numBytes = size * tfjsCore.util.bytesPerElement(dtype);
          const memoryOffset = this.wasm._malloc(numBytes);
          this.dataIdMap.set(dataId, { id, memoryOffset, shape, dtype });
          this.wasm.tfjs.registerTensor(id, size, memoryOffset);
          if (values != null) {
              this.wasm.HEAPU8.set(new Uint8Array(values.buffer, 0, numBytes), memoryOffset);
          }
      }
      async read(dataId) {
          return this.readSync(dataId);
      }
      readSync(dataId) {
          const { memoryOffset, dtype, shape, stringBytes } = this.dataIdMap.get(dataId);
          if (dtype === 'string') {
              return stringBytes;
          }
          const bytes = this.wasm.HEAPU8.slice(memoryOffset, memoryOffset + tfjsCore.util.sizeFromShape(shape) * tfjsCore.util.bytesPerElement(dtype));
          return typedArrayFromBuffer(bytes.buffer, dtype);
      }
      disposeData(dataId) {
          const data = this.dataIdMap.get(dataId);
          this.wasm._free(data.memoryOffset);
          this.wasm.tfjs.disposeData(data.id);
          this.dataIdMap.delete(dataId);
      }
      floatPrecision() {
          return 32;
      }
      // Returns the memory offset of a tensor. Useful for debugging and unit
      // testing.
      getMemoryOffset(dataId) {
          return this.dataIdMap.get(dataId).memoryOffset;
      }
      dispose() {
          this.wasm.tfjs.dispose();
          this.wasm = null;
      }
      memory() {
          return { unreliable: false };
      }
      /**
       * Make a tensor info for the output of an op. If `memoryOffset` is not
       * present, this method allocates memory on the WASM heap. If `memoryOffset`
       * is present, the memory was allocated elsewhere (in c++) and we just record
       * the pointer where that memory lives.
       */
      makeOutput(shape, dtype, memoryOffset) {
          let dataId;
          if (memoryOffset == null) {
              dataId = this.write(null /* values */, shape, dtype);
          }
          else {
              dataId = {};
              const id = this.dataIdNextNumber++;
              this.dataIdMap.set(dataId, { id, memoryOffset, shape, dtype });
              const size = tfjsCore.util.sizeFromShape(shape);
              this.wasm.tfjs.registerTensor(id, size, memoryOffset);
          }
          return { dataId, shape, dtype };
      }
      typedArrayFromHeap({ shape, dtype, dataId }) {
          const buffer = this.wasm.HEAPU8.buffer;
          const { memoryOffset } = this.dataIdMap.get(dataId);
          const size = tfjsCore.util.sizeFromShape(shape);
          switch (dtype) {
              case 'float32':
                  return new Float32Array(buffer, memoryOffset, size);
              case 'int32':
                  return new Int32Array(buffer, memoryOffset, size);
              case 'bool':
                  return new Uint8Array(buffer, memoryOffset, size);
              default:
                  throw new Error(`Uknown dtype ${dtype}`);
          }
      }
  }
  tfjsCore.registerBackend('wasm', async () => {
      const { wasm } = await init();
      return new BackendWasm(wasm);
  }, WASM_PRIORITY);
  function createInstantiateWasmFunc(path) {
      // tslint:disable-next-line:no-any
      return (imports, callback) => {
          tfjsCore.util.fetch(path, { credentials: 'same-origin' }).then((response) => {
              if (!response['ok']) {
                  imports.env.a(`failed to load wasm binary file at '${path}'`);
              }
              response.arrayBuffer().then(binary => {
                  WebAssembly.instantiate(binary, imports).then(output => {
                      callback(output.instance);
                  });
              });
          });
          return {};
      };
  }
  /**
   * Initializes the wasm module and creates the js <--> wasm bridge.
   *
   * NOTE: We wrap the wasm module in a object with property 'wasm' instead of
   * returning Promise<BackendWasmModule> to avoid freezing Chrome (last tested
   * in Chrome 76).
   */
  async function init() {
      return new Promise((resolve, reject) => {
          const factoryConfig = {};
          const locateFile = (path, prefix) => {
              if (path.endsWith('.worker.js')) {
                  const response = 'var threadInfoStruct=0;var selfThreadId=0;var parentThreadId=0;var Module={};function assert(condition,text){if(!condition)abort("Assertion failed: "+text)}function threadPrintErr(){var text=Array.prototype.slice.call(arguments).join(" ");console.error(text)}function threadAlert(){var text=Array.prototype.slice.call(arguments).join(" ");postMessage({cmd:"alert",text:text,threadId:selfThreadId})}var out=function(){throw"out() is not defined in worker.js."};var err=threadPrintErr;this.alert=threadAlert;Module["instantiateWasm"]=function(info,receiveInstance){var instance=new WebAssembly.Instance(Module["wasmModule"],info);Module["wasmModule"]=null;receiveInstance(instance);return instance.exports};this.onmessage=function(e){try{if(e.data.cmd==="load"){Module["DYNAMIC_BASE"]=e.data.DYNAMIC_BASE;Module["DYNAMICTOP_PTR"]=e.data.DYNAMICTOP_PTR;Module["wasmModule"]=e.data.wasmModule;Module["wasmMemory"]=e.data.wasmMemory;Module["buffer"]=Module["wasmMemory"].buffer;Module["ENVIRONMENT_IS_PTHREAD"]=true;if(typeof e.data.urlOrBlob==="string"){importScripts(e.data.urlOrBlob)}else{var objectUrl=URL.createObjectURL(e.data.urlOrBlob);importScripts(objectUrl);URL.revokeObjectURL(objectUrl)}Module=WasmBackendModule(Module);postMessage({"cmd":"loaded"})}else if(e.data.cmd==="objectTransfer"){Module["PThread"].receiveObjectTransfer(e.data)}else if(e.data.cmd==="run"){Module["__performance_now_clock_drift"]=performance.now()-e.data.time;threadInfoStruct=e.data.threadInfoStruct;Module["__register_pthread_ptr"](threadInfoStruct,0,0);selfThreadId=e.data.selfThreadId;parentThreadId=e.data.parentThreadId;var max=e.data.stackBase;var top=e.data.stackBase+e.data.stackSize;assert(threadInfoStruct);assert(selfThreadId);assert(parentThreadId);assert(top!=0);assert(max!=0);assert(top>max);Module["establishStackSpace"](top,max);Module["_emscripten_tls_init"]();Module["writeStackCookie"]();Module["PThread"].receiveObjectTransfer(e.data);Module["PThread"].setThreadStatus(Module["_pthread_self"](),1);try{var result=Module["dynCall_ii"](e.data.start_routine,e.data.arg);Module["checkStackCookie"]();if(!Module["getNoExitRuntime"]())Module["PThread"].threadExit(result)}catch(ex){if(ex==="Canceled!"){Module["PThread"].threadCancel()}else if(ex!="unwind"){Atomics.store(Module["HEAPU32"],threadInfoStruct+4>>2,ex instanceof Module["ExitStatus"]?ex.status:-2);Atomics.store(Module["HEAPU32"],threadInfoStruct+0>>2,1);if(typeof Module["_emscripten_futex_wake"]!=="function"){err("Thread Initialisation failed.");throw ex}Module["_emscripten_futex_wake"](threadInfoStruct+0,2147483647);if(!(ex instanceof Module["ExitStatus"]))throw ex}else{err("Pthread 0x"+threadInfoStruct.toString(16)+" completed its pthread main entry point with an unwind, keeping the pthread worker alive for asynchronous operation.")}}}else if(e.data.cmd==="cancel"){if(threadInfoStruct){Module["PThread"].threadCancel()}}else if(e.data.target==="setimmediate"){}else if(e.data.cmd==="processThreadQueue"){if(threadInfoStruct){Module["_emscripten_current_thread_process_queued_calls"]()}}else{err("worker.js received unknown command "+e.data.cmd);err(e.data)}}catch(ex){err("worker.js onmessage() captured an uncaught exception: "+ex);if(ex.stack)err(ex.stack);throw ex}};if(typeof process==="object"&&typeof process.versions==="object"&&typeof process.versions.node==="string"){self={location:{href:__filename}};var onmessage=this.onmessage;var nodeWorkerThreads=require("worker_threads");Worker=nodeWorkerThreads.Worker;var parentPort=nodeWorkerThreads.parentPort;parentPort.on("message",function(data){onmessage({data:data})});var nodeFS=require("fs");var nodeRead=function(filename){return nodeFS.readFileSync(filename,"utf8")};function globalEval(x){global.require=require;global.Module=Module;eval.call(null,x)}importScripts=function(f){globalEval(nodeRead(f))};postMessage=function(msg){parentPort.postMessage(msg)};if(typeof performance==="undefined"){performance={now:function(){return Date.now()}}}}';
                  const blob = new Blob([response], { type: 'application/javascript' });
                  return URL.createObjectURL(blob);
              }
              return prefix + path;
          };
          factoryConfig.locateFile = locateFile;
          if (wasmPath != null) {
              factoryConfig.locateFile = (path, prefix) => {
                  if (path.endsWith('.wasm')) {
                      return wasmPath;
                  }
                  return locateFile(path, prefix);
              };
              // use wasm instantiateWasm override when system fetch is not available.
              // For detail references
              // https://github.com/emscripten-core/emscripten/blob/2bca083cbbd5a4133db61fbd74d04f7feecfa907/tests/manual_wasm_instantiate.html#L170
              if (customFetch) {
                  factoryConfig.instantiateWasm = createInstantiateWasmFunc(wasmPath);
              }
          }
          const wasm = WasmBackendModule(factoryConfig);
          const voidReturnType = null;
          // Using the tfjs namespace to avoid conflict with emscripten's API.
          wasm.tfjs = {
              init: wasm.cwrap('init', null, []),
              registerTensor: wasm.cwrap('register_tensor', null, [
                  'number',
                  'number',
                  'number',
              ]),
              disposeData: wasm.cwrap('dispose_data', voidReturnType, ['number']),
              dispose: wasm.cwrap('dispose', voidReturnType, []),
          };
          let initialized = false;
          wasm.onRuntimeInitialized = () => {
              initialized = true;
              initAborted = false;
              resolve({ wasm });
          };
          wasm.onAbort = () => {
              if (initialized) {
                  // Emscripten already called console.warn so no need to double log.
                  return;
              }
              if (initAborted) {
                  // Emscripten calls `onAbort` twice, resulting in double error
                  // messages.
                  return;
              }
              initAborted = true;
              const rejectMsg = 'Make sure the server can serve the `.wasm` file relative to the ' +
                  'bundled js file. For more details see https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md#using-bundlers';
              reject({ message: rejectMsg });
          };
      });
  }
  function typedArrayFromBuffer(buffer, dtype) {
      switch (dtype) {
          case 'float32':
              return new Float32Array(buffer);
          case 'int32':
              return new Int32Array(buffer);
          case 'bool':
              return new Uint8Array(buffer);
          default:
              throw new Error(`Unknown dtype ${dtype}`);
      }
  }
  let wasmPath = null;
  let initAborted = false;
  let customFetch = false;
  /**
   * Sets the path to the `.wasm` file which will be fetched when the wasm
   * backend is initialized. See
   * https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md#using-bundlers
   * for more details.
   * @param path wasm file path or url
   * @param usePlatformFetch optional boolean to use platform fetch to download
   *     the wasm file, default to false.
   */
  /** @doc {heading: 'Environment', namespace: 'wasm'} */
  function setWasmPath(path, usePlatformFetch = false) {
      if (initAborted) {
          throw new Error('The WASM backend was already initialized. Make sure you call ' +
              '`setWasmPath()` before you call `tf.setBackend()` or `tf.ready()`');
      }
      wasmPath = path;
      customFetch = usePlatformFetch;
  }

  /** @license See the LICENSE file. */
  // This code is auto-generated, do not modify this file!
  const version = '0.0.0';

  exports.BackendWasm = BackendWasm;
  exports.setWasmPath = setWasmPath;
  exports.version_wasm = version;

  Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=tf-backend-wasm.js.map
