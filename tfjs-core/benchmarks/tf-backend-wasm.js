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
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('@tensorflow/tfjs-core'), require('path'), require('fs')) :
  typeof define === 'function' && define.amd ? define(['exports', '@tensorflow/tfjs-core', 'path', 'fs'], factory) :
  (global = global || self, factory((global.tf = global.tf || {}, global.tf.wasm = global.tf.wasm || {}), global.tf, global.path, global.fs));
}(this, (function (exports, tfjsCore, path, fs) { 'use strict';

  path = path && path.hasOwnProperty('default') ? path['default'] : path;
  fs = fs && fs.hasOwnProperty('default') ? fs['default'] : fs;

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
      const x = inputs.x;
      const xId = backend.dataIdMap.get(x.dataId).id;
      const { filterSize, strides, pad, dimRoundingMode } = attrs;
      const convInfo = tfjsCore.backend_util.computePool2DInfo(x.shape, filterSize, strides, 1 /* dilations */, pad, dimRoundingMode);
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
      const x = inputs.x;
      const xId = backend.dataIdMap.get(x.dataId).id;
      const { filterSize, strides, pad, dimRoundingMode } = attrs;
      const convInfo = tfjsCore.backend_util.computePool2DInfo(x.shape, filterSize, strides, 1 /* dilations */, pad, dimRoundingMode);
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
  const ENV = tfjsCore.env();
  /**
   * True if SIMD is supported.
   */
  // From: https://github.com/GoogleChromeLabs/wasm-feature-detect
  ENV.registerFlag('WASM_HAS_SIMD_SUPPORT', async () => WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3,
      2, 1, 0, 10, 9, 1, 7, 0, 65, 0, 253, 15, 26, 11
  ])));

  function createCommonjsModule(fn, module) {
  	return module = { exports: {} }, fn(module, module.exports), module.exports;
  }

  var tfjsBackendWasm = createCommonjsModule(function (module, exports) {
  var WasmBackendModule = (function() {
    var _scriptDir = typeof document !== 'undefined' && document.currentScript ? document.currentScript.src : undefined;
    if (typeof __filename !== 'undefined') _scriptDir = _scriptDir || __filename;
    return (
  function(WasmBackendModule) {
    WasmBackendModule = WasmBackendModule || {};

  var Module=typeof WasmBackendModule!=="undefined"?WasmBackendModule:{};var moduleOverrides={};var key;for(key in Module){if(Module.hasOwnProperty(key)){moduleOverrides[key]=Module[key];}}var arguments_=[];var thisProgram="./this.program";var quit_=function(status,toThrow){throw toThrow};var ENVIRONMENT_IS_WEB=false;var ENVIRONMENT_IS_WORKER=false;var ENVIRONMENT_IS_NODE=false;var ENVIRONMENT_IS_SHELL=false;ENVIRONMENT_IS_WEB=typeof window==="object";ENVIRONMENT_IS_WORKER=typeof importScripts==="function";ENVIRONMENT_IS_NODE=typeof process==="object"&&typeof process.versions==="object"&&typeof process.versions.node==="string";ENVIRONMENT_IS_SHELL=!ENVIRONMENT_IS_WEB&&!ENVIRONMENT_IS_NODE&&!ENVIRONMENT_IS_WORKER;var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var read_,readBinary;var nodeFS;var nodePath;if(ENVIRONMENT_IS_NODE){if(ENVIRONMENT_IS_WORKER){scriptDirectory=path.dirname(scriptDirectory)+"/";}else{scriptDirectory=__dirname+"/";}read_=function shell_read(filename,binary){if(!nodeFS)nodeFS=fs;if(!nodePath)nodePath=path;filename=nodePath["normalize"](filename);return nodeFS["readFileSync"](filename,binary?null:"utf8")};readBinary=function readBinary(filename){var ret=read_(filename,true);if(!ret.buffer){ret=new Uint8Array(ret);}assert(ret.buffer);return ret};if(process["argv"].length>1){thisProgram=process["argv"][1].replace(/\\/g,"/");}arguments_=process["argv"].slice(2);process["on"]("uncaughtException",function(ex){if(!(ex instanceof ExitStatus)){throw ex}});process["on"]("unhandledRejection",abort);quit_=function(status){process["exit"](status);};Module["inspect"]=function(){return "[Emscripten Module object]"};}else if(ENVIRONMENT_IS_SHELL){if(typeof read!="undefined"){read_=function shell_read(f){return read(f)};}readBinary=function readBinary(f){var data;if(typeof readbuffer==="function"){return new Uint8Array(readbuffer(f))}data=read(f,"binary");assert(typeof data==="object");return data};if(typeof scriptArgs!="undefined"){arguments_=scriptArgs;}else if(typeof arguments!="undefined"){arguments_=arguments;}if(typeof quit==="function"){quit_=function(status){quit(status);};}if(typeof print!=="undefined"){if(typeof console==="undefined")console={};console.log=print;console.warn=console.error=typeof printErr!=="undefined"?printErr:print;}}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){if(ENVIRONMENT_IS_WORKER){scriptDirectory=self.location.href;}else if(document.currentScript){scriptDirectory=document.currentScript.src;}if(_scriptDir){scriptDirectory=_scriptDir;}if(scriptDirectory.indexOf("blob:")!==0){scriptDirectory=scriptDirectory.substr(0,scriptDirectory.lastIndexOf("/")+1);}else{scriptDirectory="";}{read_=function shell_read(url){var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.send(null);return xhr.responseText};if(ENVIRONMENT_IS_WORKER){readBinary=function readBinary(url){var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)};}}}var out=Module["print"]||console.log.bind(console);var err=Module["printErr"]||console.warn.bind(console);for(key in moduleOverrides){if(moduleOverrides.hasOwnProperty(key)){Module[key]=moduleOverrides[key];}}moduleOverrides=null;if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"];if(Module["quit"])quit_=Module["quit"];var wasmBinary;if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];var noExitRuntime;if(Module["noExitRuntime"])noExitRuntime=Module["noExitRuntime"];if(typeof WebAssembly!=="object"){err("no native wasm support detected");}var wasmMemory;var wasmTable=new WebAssembly.Table({"initial":124,"maximum":124+0,"element":"anyfunc"});var ABORT=false;function assert(condition,text){if(!condition){abort("Assertion failed: "+text);}}function getCFunc(ident){var func=Module["_"+ident];assert(func,"Cannot call unknown function "+ident+", make sure it is exported");return func}function ccall(ident,returnType,argTypes,args,opts){var toC={"string":function(str){var ret=0;if(str!==null&&str!==undefined&&str!==0){var len=(str.length<<2)+1;ret=stackAlloc(len);stringToUTF8(str,ret,len);}return ret},"array":function(arr){var ret=stackAlloc(arr.length);writeArrayToMemory(arr,ret);return ret}};function convertReturnValue(ret){if(returnType==="string")return UTF8ToString(ret);if(returnType==="boolean")return Boolean(ret);return ret}var func=getCFunc(ident);var cArgs=[];var stack=0;if(args){for(var i=0;i<args.length;i++){var converter=toC[argTypes[i]];if(converter){if(stack===0)stack=stackSave();cArgs[i]=converter(args[i]);}else{cArgs[i]=args[i];}}}var ret=func.apply(null,cArgs);ret=convertReturnValue(ret);if(stack!==0)stackRestore(stack);return ret}function cwrap(ident,returnType,argTypes,opts){argTypes=argTypes||[];var numericArgs=argTypes.every(function(type){return type==="number"});var numericRet=returnType!=="string";if(numericRet&&numericArgs&&!opts){return getCFunc(ident)}return function(){return ccall(ident,returnType,argTypes,arguments)}}var UTF8Decoder=typeof TextDecoder!=="undefined"?new TextDecoder("utf8"):undefined;function UTF8ArrayToString(heap,idx,maxBytesToRead){var endIdx=idx+maxBytesToRead;var endPtr=idx;while(heap[endPtr]&&!(endPtr>=endIdx))++endPtr;if(endPtr-idx>16&&heap.subarray&&UTF8Decoder){return UTF8Decoder.decode(heap.subarray(idx,endPtr))}else{var str="";while(idx<endPtr){var u0=heap[idx++];if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=heap[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=heap[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2;}else{u0=(u0&7)<<18|u1<<12|u2<<6|heap[idx++]&63;}if(u0<65536){str+=String.fromCharCode(u0);}else{var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023);}}}return str}function UTF8ToString(ptr,maxBytesToRead){return ptr?UTF8ArrayToString(HEAPU8,ptr,maxBytesToRead):""}function stringToUTF8Array(str,heap,outIdx,maxBytesToWrite){if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.charCodeAt(i);if(u>=55296&&u<=57343){var u1=str.charCodeAt(++i);u=65536+((u&1023)<<10)|u1&1023;}if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u;}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63;}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;}else{if(outIdx+3>=endIdx)break;heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;}}heap[outIdx]=0;return outIdx-startIdx}function stringToUTF8(str,outPtr,maxBytesToWrite){return stringToUTF8Array(str,HEAPU8,outPtr,maxBytesToWrite)}var UTF16Decoder=typeof TextDecoder!=="undefined"?new TextDecoder("utf-16le"):undefined;function writeArrayToMemory(array,buffer){HEAP8.set(array,buffer);}var WASM_PAGE_SIZE=65536;function alignUp(x,multiple){if(x%multiple>0){x+=multiple-x%multiple;}return x}var buffer,HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;function updateGlobalBufferAndViews(buf){buffer=buf;Module["HEAP8"]=HEAP8=new Int8Array(buf);Module["HEAP16"]=HEAP16=new Int16Array(buf);Module["HEAP32"]=HEAP32=new Int32Array(buf);Module["HEAPU8"]=HEAPU8=new Uint8Array(buf);Module["HEAPU16"]=HEAPU16=new Uint16Array(buf);Module["HEAPU32"]=HEAPU32=new Uint32Array(buf);Module["HEAPF32"]=HEAPF32=new Float32Array(buf);Module["HEAPF64"]=HEAPF64=new Float64Array(buf);}var DYNAMIC_BASE=5254224,DYNAMICTOP_PTR=11184;var INITIAL_INITIAL_MEMORY=Module["INITIAL_MEMORY"]||16777216;if(Module["wasmMemory"]){wasmMemory=Module["wasmMemory"];}else{wasmMemory=new WebAssembly.Memory({"initial":INITIAL_INITIAL_MEMORY/WASM_PAGE_SIZE,"maximum":2147483648/WASM_PAGE_SIZE});}if(wasmMemory){buffer=wasmMemory.buffer;}INITIAL_INITIAL_MEMORY=buffer.byteLength;updateGlobalBufferAndViews(buffer);HEAP32[DYNAMICTOP_PTR>>2]=DYNAMIC_BASE;function callRuntimeCallbacks(callbacks){while(callbacks.length>0){var callback=callbacks.shift();if(typeof callback=="function"){callback(Module);continue}var func=callback.func;if(typeof func==="number"){if(callback.arg===undefined){Module["dynCall_v"](func);}else{Module["dynCall_vi"](func,callback.arg);}}else{func(callback.arg===undefined?null:callback.arg);}}}var __ATPRERUN__=[];var __ATINIT__=[];var __ATMAIN__=[];var __ATPOSTRUN__=[];function preRun(){if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift());}}callRuntimeCallbacks(__ATPRERUN__);}function initRuntime(){callRuntimeCallbacks(__ATINIT__);}function preMain(){callRuntimeCallbacks(__ATMAIN__);}function postRun(){if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift());}}callRuntimeCallbacks(__ATPOSTRUN__);}function addOnPreRun(cb){__ATPRERUN__.unshift(cb);}function addOnPostRun(cb){__ATPOSTRUN__.unshift(cb);}var Math_ceil=Math.ceil;var Math_floor=Math.floor;var runDependencies=0;var runDependencyWatcher=null;var dependenciesFulfilled=null;function addRunDependency(id){runDependencies++;if(Module["monitorRunDependencies"]){Module["monitorRunDependencies"](runDependencies);}}function removeRunDependency(id){runDependencies--;if(Module["monitorRunDependencies"]){Module["monitorRunDependencies"](runDependencies);}if(runDependencies==0){if(runDependencyWatcher!==null){clearInterval(runDependencyWatcher);runDependencyWatcher=null;}if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback();}}}Module["preloadedImages"]={};Module["preloadedAudios"]={};function abort(what){if(Module["onAbort"]){Module["onAbort"](what);}what+="";out(what);err(what);ABORT=true;what="abort("+what+"). Build with -s ASSERTIONS=1 for more info.";throw new WebAssembly.RuntimeError(what)}function hasPrefix(str,prefix){return String.prototype.startsWith?str.startsWith(prefix):str.indexOf(prefix)===0}var dataURIPrefix="data:application/octet-stream;base64,";function isDataURI(filename){return hasPrefix(filename,dataURIPrefix)}var fileURIPrefix="file://";function isFileURI(filename){return hasPrefix(filename,fileURIPrefix)}var wasmBinaryFile="tfjs-backend-wasm.wasm";if(!isDataURI(wasmBinaryFile)){wasmBinaryFile=locateFile(wasmBinaryFile);}function getBinary(){try{if(wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(wasmBinaryFile)}else{throw "both async and sync fetching of the wasm failed"}}catch(err){abort(err);}}function getBinaryPromise(){if(!wasmBinary&&(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER)&&typeof fetch==="function"&&!isFileURI(wasmBinaryFile)){return fetch(wasmBinaryFile,{credentials:"same-origin"}).then(function(response){if(!response["ok"]){throw "failed to load wasm binary file at '"+wasmBinaryFile+"'"}return response["arrayBuffer"]()}).catch(function(){return getBinary()})}return new Promise(function(resolve,reject){resolve(getBinary());})}function createWasm(){var info={"a":asmLibraryArg};function receiveInstance(instance,module){var exports=instance.exports;Module["asm"]=exports;removeRunDependency();}addRunDependency();function receiveInstantiatedSource(output){receiveInstance(output["instance"]);}function instantiateArrayBuffer(receiver){return getBinaryPromise().then(function(binary){return WebAssembly.instantiate(binary,info)}).then(receiver,function(reason){err("failed to asynchronously prepare wasm: "+reason);abort(reason);})}function instantiateAsync(){if(!wasmBinary&&typeof WebAssembly.instantiateStreaming==="function"&&!isDataURI(wasmBinaryFile)&&!isFileURI(wasmBinaryFile)&&typeof fetch==="function"){fetch(wasmBinaryFile,{credentials:"same-origin"}).then(function(response){var result=WebAssembly.instantiateStreaming(response,info);return result.then(receiveInstantiatedSource,function(reason){err("wasm streaming compile failed: "+reason);err("falling back to ArrayBuffer instantiation");instantiateArrayBuffer(receiveInstantiatedSource);})});}else{return instantiateArrayBuffer(receiveInstantiatedSource)}}if(Module["instantiateWasm"]){try{var exports=Module["instantiateWasm"](info,receiveInstance);return exports}catch(e){err("Module.instantiateWasm callback failed with error: "+e);return false}}instantiateAsync();return {}}__ATINIT__.push({func:function(){___wasm_call_ctors();}});function _abort(){abort();}function _emscripten_memcpy_big(dest,src,num){HEAPU8.copyWithin(dest,src,src+num);}function _emscripten_get_heap_size(){return HEAPU8.length}function emscripten_realloc_buffer(size){try{wasmMemory.grow(size-buffer.byteLength+65535>>>16);updateGlobalBufferAndViews(wasmMemory.buffer);return 1}catch(e){}}function _emscripten_resize_heap(requestedSize){var oldSize=_emscripten_get_heap_size();var PAGE_MULTIPLE=65536;var maxHeapSize=2147483648;if(requestedSize>maxHeapSize){return false}var minHeapSize=16777216;for(var cutDown=1;cutDown<=4;cutDown*=2){var overGrownHeapSize=oldSize*(1+.2/cutDown);overGrownHeapSize=Math.min(overGrownHeapSize,requestedSize+100663296);var newSize=Math.min(maxHeapSize,alignUp(Math.max(minHeapSize,requestedSize,overGrownHeapSize),PAGE_MULTIPLE));var replacement=emscripten_realloc_buffer(newSize);if(replacement){return true}}return false}var SYSCALLS={mappings:{},buffers:[null,[],[]],printChar:function(stream,curr){var buffer=SYSCALLS.buffers[stream];if(curr===0||curr===10){(stream===1?out:err)(UTF8ArrayToString(buffer,0));buffer.length=0;}else{buffer.push(curr);}},varargs:undefined,get:function(){SYSCALLS.varargs+=4;var ret=HEAP32[SYSCALLS.varargs-4>>2];return ret},getStr:function(ptr){var ret=UTF8ToString(ptr);return ret},get64:function(low,high){return low}};function _fd_close(fd){return 0}function _fd_seek(fd,offset_low,offset_high,whence,newOffset){}function _fd_write(fd,iov,iovcnt,pnum){var num=0;for(var i=0;i<iovcnt;i++){var ptr=HEAP32[iov+i*8>>2];var len=HEAP32[iov+(i*8+4)>>2];for(var j=0;j<len;j++){SYSCALLS.printChar(fd,HEAPU8[ptr+j]);}num+=len;}HEAP32[pnum>>2]=num;return 0}function _roundf(d){d=+d;return d>=+0?+Math_floor(d+ +.5):+Math_ceil(d-+.5)}var asmLibraryArg={"a":_abort,"e":_emscripten_memcpy_big,"f":_emscripten_resize_heap,"g":_fd_close,"d":_fd_seek,"c":_fd_write,"memory":wasmMemory,"b":_roundf,"table":wasmTable};var asm=createWasm();Module["asm"]=asm;var ___wasm_call_ctors=Module["___wasm_call_ctors"]=function(){return (___wasm_call_ctors=Module["___wasm_call_ctors"]=Module["asm"]["h"]).apply(null,arguments)};var _init=Module["_init"]=function(){return (_init=Module["_init"]=Module["asm"]["i"]).apply(null,arguments)};var _register_tensor=Module["_register_tensor"]=function(){return (_register_tensor=Module["_register_tensor"]=Module["asm"]["j"]).apply(null,arguments)};var _dispose_data=Module["_dispose_data"]=function(){return (_dispose_data=Module["_dispose_data"]=Module["asm"]["k"]).apply(null,arguments)};var _dispose=Module["_dispose"]=function(){return (_dispose=Module["_dispose"]=Module["asm"]["l"]).apply(null,arguments)};var _Abs=Module["_Abs"]=function(){return (_Abs=Module["_Abs"]=Module["asm"]["m"]).apply(null,arguments)};var _Add=Module["_Add"]=function(){return (_Add=Module["_Add"]=Module["asm"]["n"]).apply(null,arguments)};var _AddN=Module["_AddN"]=function(){return (_AddN=Module["_AddN"]=Module["asm"]["o"]).apply(null,arguments)};var _ArgMax=Module["_ArgMax"]=function(){return (_ArgMax=Module["_ArgMax"]=Module["asm"]["p"]).apply(null,arguments)};var _AvgPool=Module["_AvgPool"]=function(){return (_AvgPool=Module["_AvgPool"]=Module["asm"]["q"]).apply(null,arguments)};var _BatchMatMul=Module["_BatchMatMul"]=function(){return (_BatchMatMul=Module["_BatchMatMul"]=Module["asm"]["r"]).apply(null,arguments)};var _ClipByValue=Module["_ClipByValue"]=function(){return (_ClipByValue=Module["_ClipByValue"]=Module["asm"]["s"]).apply(null,arguments)};var _Conv2D=Module["_Conv2D"]=function(){return (_Conv2D=Module["_Conv2D"]=Module["asm"]["t"]).apply(null,arguments)};var _Cos=Module["_Cos"]=function(){return (_Cos=Module["_Cos"]=Module["asm"]["u"]).apply(null,arguments)};var _CropAndResize=Module["_CropAndResize"]=function(){return (_CropAndResize=Module["_CropAndResize"]=Module["asm"]["v"]).apply(null,arguments)};var _DepthwiseConv2dNative=Module["_DepthwiseConv2dNative"]=function(){return (_DepthwiseConv2dNative=Module["_DepthwiseConv2dNative"]=Module["asm"]["w"]).apply(null,arguments)};var _Div=Module["_Div"]=function(){return (_Div=Module["_Div"]=Module["asm"]["x"]).apply(null,arguments)};var _Exp=Module["_Exp"]=function(){return (_Exp=Module["_Exp"]=Module["asm"]["y"]).apply(null,arguments)};var _FloorDiv=Module["_FloorDiv"]=function(){return (_FloorDiv=Module["_FloorDiv"]=Module["asm"]["z"]).apply(null,arguments)};var _FusedBatchNorm=Module["_FusedBatchNorm"]=function(){return (_FusedBatchNorm=Module["_FusedBatchNorm"]=Module["asm"]["A"]).apply(null,arguments)};var _FusedConv2D=Module["_FusedConv2D"]=function(){return (_FusedConv2D=Module["_FusedConv2D"]=Module["asm"]["B"]).apply(null,arguments)};var _FusedDepthwiseConv2D=Module["_FusedDepthwiseConv2D"]=function(){return (_FusedDepthwiseConv2D=Module["_FusedDepthwiseConv2D"]=Module["asm"]["C"]).apply(null,arguments)};var _Gather=Module["_Gather"]=function(){return (_Gather=Module["_Gather"]=Module["asm"]["D"]).apply(null,arguments)};var _GatherNd=Module["_GatherNd"]=function(){return (_GatherNd=Module["_GatherNd"]=Module["asm"]["E"]).apply(null,arguments)};var _Greater=Module["_Greater"]=function(){return (_Greater=Module["_Greater"]=Module["asm"]["F"]).apply(null,arguments)};var _GreaterEqual=Module["_GreaterEqual"]=function(){return (_GreaterEqual=Module["_GreaterEqual"]=Module["asm"]["G"]).apply(null,arguments)};var _Less=Module["_Less"]=function(){return (_Less=Module["_Less"]=Module["asm"]["H"]).apply(null,arguments)};var _LessEqual=Module["_LessEqual"]=function(){return (_LessEqual=Module["_LessEqual"]=Module["asm"]["I"]).apply(null,arguments)};var _Log=Module["_Log"]=function(){return (_Log=Module["_Log"]=Module["asm"]["J"]).apply(null,arguments)};var _LogicalAnd=Module["_LogicalAnd"]=function(){return (_LogicalAnd=Module["_LogicalAnd"]=Module["asm"]["K"]).apply(null,arguments)};var _Max=Module["_Max"]=function(){return (_Max=Module["_Max"]=Module["asm"]["L"]).apply(null,arguments)};var _MaxPool=Module["_MaxPool"]=function(){return (_MaxPool=Module["_MaxPool"]=Module["asm"]["M"]).apply(null,arguments)};var _Maximum=Module["_Maximum"]=function(){return (_Maximum=Module["_Maximum"]=Module["asm"]["N"]).apply(null,arguments)};var _Min=Module["_Min"]=function(){return (_Min=Module["_Min"]=Module["asm"]["O"]).apply(null,arguments)};var _Minimum=Module["_Minimum"]=function(){return (_Minimum=Module["_Minimum"]=Module["asm"]["P"]).apply(null,arguments)};var _Mul=Module["_Mul"]=function(){return (_Mul=Module["_Mul"]=Module["asm"]["Q"]).apply(null,arguments)};var _Neg=Module["_Neg"]=function(){return (_Neg=Module["_Neg"]=Module["asm"]["R"]).apply(null,arguments)};var _NonMaxSuppressionV3=Module["_NonMaxSuppressionV3"]=function(){return (_NonMaxSuppressionV3=Module["_NonMaxSuppressionV3"]=Module["asm"]["S"]).apply(null,arguments)};var _NonMaxSuppressionV5=Module["_NonMaxSuppressionV5"]=function(){return (_NonMaxSuppressionV5=Module["_NonMaxSuppressionV5"]=Module["asm"]["T"]).apply(null,arguments)};var _NotEqual=Module["_NotEqual"]=function(){return (_NotEqual=Module["_NotEqual"]=Module["asm"]["U"]).apply(null,arguments)};var _PadV2=Module["_PadV2"]=function(){return (_PadV2=Module["_PadV2"]=Module["asm"]["V"]).apply(null,arguments)};var _Pow=Module["_Pow"]=function(){return (_Pow=Module["_Pow"]=Module["asm"]["W"]).apply(null,arguments)};var _Prelu=Module["_Prelu"]=function(){return (_Prelu=Module["_Prelu"]=Module["asm"]["X"]).apply(null,arguments)};var _Relu=Module["_Relu"]=function(){return (_Relu=Module["_Relu"]=Module["asm"]["Y"]).apply(null,arguments)};var _Relu6=Module["_Relu6"]=function(){return (_Relu6=Module["_Relu6"]=Module["asm"]["Z"]).apply(null,arguments)};var _ResizeBilinear=Module["_ResizeBilinear"]=function(){return (_ResizeBilinear=Module["_ResizeBilinear"]=Module["asm"]["_"]).apply(null,arguments)};var _Rsqrt=Module["_Rsqrt"]=function(){return (_Rsqrt=Module["_Rsqrt"]=Module["asm"]["$"]).apply(null,arguments)};var _ScatterNd=Module["_ScatterNd"]=function(){return (_ScatterNd=Module["_ScatterNd"]=Module["asm"]["aa"]).apply(null,arguments)};var _Sigmoid=Module["_Sigmoid"]=function(){return (_Sigmoid=Module["_Sigmoid"]=Module["asm"]["ba"]).apply(null,arguments)};var _Sin=Module["_Sin"]=function(){return (_Sin=Module["_Sin"]=Module["asm"]["ca"]).apply(null,arguments)};var _Softmax=Module["_Softmax"]=function(){return (_Softmax=Module["_Softmax"]=Module["asm"]["da"]).apply(null,arguments)};var _Sqrt=Module["_Sqrt"]=function(){return (_Sqrt=Module["_Sqrt"]=Module["asm"]["ea"]).apply(null,arguments)};var _Square=Module["_Square"]=function(){return (_Square=Module["_Square"]=Module["asm"]["fa"]).apply(null,arguments)};var _Sub=Module["_Sub"]=function(){return (_Sub=Module["_Sub"]=Module["asm"]["ga"]).apply(null,arguments)};var _Sum=Module["_Sum"]=function(){return (_Sum=Module["_Sum"]=Module["asm"]["ha"]).apply(null,arguments)};var _Tanh=Module["_Tanh"]=function(){return (_Tanh=Module["_Tanh"]=Module["asm"]["ia"]).apply(null,arguments)};var _Tile=Module["_Tile"]=function(){return (_Tile=Module["_Tile"]=Module["asm"]["ja"]).apply(null,arguments)};var _Transpose=Module["_Transpose"]=function(){return (_Transpose=Module["_Transpose"]=Module["asm"]["ka"]).apply(null,arguments)};var __FusedMatMul=Module["__FusedMatMul"]=function(){return (__FusedMatMul=Module["__FusedMatMul"]=Module["asm"]["la"]).apply(null,arguments)};var _malloc=Module["_malloc"]=function(){return (_malloc=Module["_malloc"]=Module["asm"]["ma"]).apply(null,arguments)};var _free=Module["_free"]=function(){return (_free=Module["_free"]=Module["asm"]["na"]).apply(null,arguments)};var stackSave=Module["stackSave"]=function(){return (stackSave=Module["stackSave"]=Module["asm"]["oa"]).apply(null,arguments)};var stackAlloc=Module["stackAlloc"]=function(){return (stackAlloc=Module["stackAlloc"]=Module["asm"]["pa"]).apply(null,arguments)};var stackRestore=Module["stackRestore"]=function(){return (stackRestore=Module["stackRestore"]=Module["asm"]["qa"]).apply(null,arguments)};var dynCall_vi=Module["dynCall_vi"]=function(){return (dynCall_vi=Module["dynCall_vi"]=Module["asm"]["ra"]).apply(null,arguments)};var dynCall_v=Module["dynCall_v"]=function(){return (dynCall_v=Module["dynCall_v"]=Module["asm"]["sa"]).apply(null,arguments)};Module["asm"]=asm;Module["cwrap"]=cwrap;var calledRun;Module["then"]=function(func){if(calledRun){func(Module);}else{var old=Module["onRuntimeInitialized"];Module["onRuntimeInitialized"]=function(){if(old)old();func(Module);};}return Module};function ExitStatus(status){this.name="ExitStatus";this.message="Program terminated with exit("+status+")";this.status=status;}dependenciesFulfilled=function runCaller(){if(!calledRun)run();if(!calledRun)dependenciesFulfilled=runCaller;};function run(args){if(runDependencies>0){return}preRun();if(runDependencies>0)return;function doRun(){if(calledRun)return;calledRun=true;Module["calledRun"]=true;if(ABORT)return;initRuntime();preMain();if(Module["onRuntimeInitialized"])Module["onRuntimeInitialized"]();postRun();}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(function(){setTimeout(function(){Module["setStatus"]("");},1);doRun();},1);}else{doRun();}}Module["run"]=run;if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].pop()();}}noExitRuntime=true;run();


    return WasmBackendModule
  }
  );
  })();
  module.exports = WasmBackendModule;
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
              this.wasm.HEAPU8.set(new Uint8Array(values.buffer, values.byteOffset, numBytes), memoryOffset);
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
      const simdSupported = await tfjsCore.env().getAsync('WASM_HAS_SIMD_SUPPORT');
      return new Promise((resolve, reject) => {
          const factoryConfig = {};
          if (wasmPath != null) {
              factoryConfig.locateFile = (path, prefix) => {
                  if (path.endsWith('.wasm')) {
                      console.log('SETTING WASM BINARY LOCATION.', simdSupported);
                      return wasmPath;
                  }
                  return prefix + path;
              };
              // use wasm instantiateWasm override when system fetch is not available.
              // For detail references
              // https://github.com/emscripten-core/emscripten/blob/2bca083cbbd5a4133db61fbd74d04f7feecfa907/tests/manual_wasm_instantiate.html#L170
              if (customFetch) {
                  factoryConfig.instantiateWasm = createInstantiateWasmFunc(wasmPath);
              }
          }
          const wasm = tfjsBackendWasm(factoryConfig);
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
