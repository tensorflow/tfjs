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
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('@tensorflow/tfjs-core'), require('path'), require('fs'), require('worker_threads'), require('perf_hooks')) :
  typeof define === 'function' && define.amd ? define(['exports', '@tensorflow/tfjs-core', 'path', 'fs', 'worker_threads', 'perf_hooks'], factory) :
  (global = global || self, factory((global.tf = global.tf || {}, global.tf.wasm = global.tf.wasm || {}), global.tf, global.path, global.fs, global.worker_threads, global.perf_hooks));
}(this, (function (exports, tfjsCore, path, fs, worker_threads, perf_hooks) { 'use strict';

  path = path && path.hasOwnProperty('default') ? path['default'] : path;
  fs = fs && fs.hasOwnProperty('default') ? fs['default'] : fs;
  worker_threads = worker_threads && worker_threads.hasOwnProperty('default') ? worker_threads['default'] : worker_threads;
  perf_hooks = perf_hooks && perf_hooks.hasOwnProperty('default') ? perf_hooks['default'] : perf_hooks;

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
  var wasmFusedMatMul;
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
      var inputs = args.inputs, backend = args.backend, attrs = args.attrs;
      var a = inputs.a, b = inputs.b, bias = inputs.bias, preluActivationWeights = inputs.preluActivationWeights;
      if (a.dtype !== 'float32' || b.dtype !== 'float32') {
          throw new Error("_FusedMatMul for non non-float32 tensors not yet supported.");
      }
      var transposeA = attrs.transposeA, transposeB = attrs.transposeB, activation = attrs.activation;
      var aId = backend.dataIdMap.get(a.dataId).id;
      var bId = backend.dataIdMap.get(b.dataId).id;
      var biasId = 0;
      if (bias != null) {
          var biasData = backend.dataIdMap.get(bias.dataId);
          if (biasData.shape.length !== 1) {
              throw new Error("_FusedMatMul only supports rank-1 bias but got " +
                  ("rank " + biasData.shape.length + "."));
          }
          biasId = biasData.id;
      }
      var preluActivationWeightsId = preluActivationWeights == null ?
          0 :
          backend.dataIdMap.get(preluActivationWeights.dataId).id;
      var fusedActivation = FusableActivation[activation];
      if (fusedActivation == null) {
          throw new Error(activation + " activation not yet supported for FusedConv2D " +
              "in the wasm backend.");
      }
      var leftDim = transposeA ? a.shape[2] : a.shape[1];
      var rightDim = transposeB ? b.shape[1] : b.shape[2];
      var batchDim = a.shape[0];
      var out = backend.makeOutput([batchDim, leftDim, rightDim], a.dtype);
      var outId = backend.dataIdMap.get(out.dataId).id;
      var aShapeBytes = new Uint8Array(new Int32Array(a.shape).buffer);
      var bShapeBytes = new Uint8Array(new Int32Array(b.shape).buffer);
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
      var wasmFunc;
      function setupFunc(backend) {
          wasmFunc =
              backend.wasm.cwrap(kernelName, null /* void */, ['number', 'number']);
      }
      function kernelFunc(args) {
          var backend = args.backend, x = args.inputs.x;
          var xId = backend.dataIdMap.get(x.dataId).id;
          var out = backend.makeOutput(x.shape, x.dtype);
          var outId = backend.dataIdMap.get(out.dataId).id;
          // Short-circuit zero-sized tensors.
          if (tfjsCore.util.sizeFromShape(out.shape) === 0) {
              return out;
          }
          wasmFunc(xId, outId);
          return out;
      }
      tfjsCore.registerKernel({ kernelName: kernelName, backendName: 'wasm', setupFunc: setupFunc, kernelFunc: kernelFunc });
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
      var wasmFunc;
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
          var backend = args.backend, inputs = args.inputs;
          var a = inputs.a, b = inputs.b;
          var aId = backend.dataIdMap.get(a.dataId).id;
          var bId = backend.dataIdMap.get(b.dataId).id;
          var outputType = dtype != null ? dtype : a.dtype;
          var newShape = tfjsCore.backend_util.assertAndGetBroadcastShape(a.shape, b.shape);
          var out = backend.makeOutput(newShape, outputType);
          // Short-circuit zero-sized tensors.
          if (tfjsCore.util.sizeFromShape(newShape) === 0) {
              return out;
          }
          var aShapeBytes = new Uint8Array(new Int32Array(a.shape).buffer);
          var bShapeBytes = new Uint8Array(new Int32Array(b.shape).buffer);
          var outId = backend.dataIdMap.get(out.dataId).id;
          var kernelFunc = function () { return wasmFunc(aId, aShapeBytes, a.shape.length, bId, bShapeBytes, b.shape.length, CppDType[a.dtype], outId); };
          if (supportsFullBroadcast) {
              kernelFunc();
              return out;
          }
          var aBroadcastDims = tfjsCore.backend_util.getBroadcastDims(a.shape, newShape);
          var bBroadcastDims = tfjsCore.backend_util.getBroadcastDims(b.shape, newShape);
          var loopsOverAllOfA = aBroadcastDims.every(function (v, i) { return v === i; });
          var loopsOverAllOfB = bBroadcastDims.every(function (v, i) { return v === i; });
          if (loopsOverAllOfA && loopsOverAllOfB) {
              kernelFunc();
              return out;
          }
          else {
              throw new Error("Broadcasting along outer dims is not yet " +
                  ("supported for " + kernelName + "."));
          }
      }
      tfjsCore.registerKernel({ kernelName: kernelName, backendName: 'wasm', setupFunc: setupFunc, kernelFunc: kernelFunc });
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
  var supportsFullBroadcast = true;
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
  var wasmFunc;
  function setupFunc(backend) {
      wasmFunc = backend.wasm.cwrap('AddN', null /* void */, [
          'array',
          'number',
          'number',
          'number',
      ]);
  }
  function addn(args) {
      var inputs = args.inputs, backend = args.backend;
      var out = backend.makeOutput(inputs[0].shape, inputs[0].dtype);
      // Short-circuit zero-sized tensors.
      if (tfjsCore.util.sizeFromShape(out.shape) === 0) {
          return out;
      }
      var inputIds = inputs.map(function (x) { return backend.dataIdMap.get(x.dataId).id; });
      var inputIdsBytes = new Uint8Array(new Int32Array(inputIds).buffer);
      var outId = backend.dataIdMap.get(out.dataId).id;
      wasmFunc(inputIdsBytes, inputIds.length, CppDType[out.dtype], outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'AddN',
      backendName: 'wasm',
      setupFunc: setupFunc,
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
  var wasmFunc$1;
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
      var x = args.inputs.x, backend = args.backend, axis = args.attrs.axis;
      var outShape = x.shape.slice(0, -1);
      var out = backend.makeOutput(outShape, 'int32');
      var xId = backend.dataIdMap.get(x.dataId).id;
      var outId = backend.dataIdMap.get(out.dataId).id;
      var outerSize = tfjsCore.util.sizeFromShape(out.shape);
      var innerSize = x.shape[axis];
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
  var wasmAvgPool;
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
      var inputs = args.inputs, attrs = args.attrs, backend = args.backend;
      var convInfo = attrs;
      var x = inputs.x;
      var xId = backend.dataIdMap.get(x.dataId).id;
      var filterHeight = convInfo.filterHeight;
      var filterWidth = convInfo.filterWidth;
      var padTop = convInfo.padInfo.top;
      var padRight = convInfo.padInfo.right;
      var padBottom = convInfo.padInfo.bottom;
      var padLeft = convInfo.padInfo.left;
      var strideHeight = convInfo.strideHeight;
      var strideWidth = convInfo.strideWidth;
      var channels = convInfo.inChannels;
      if (convInfo.dataFormat !== 'channelsLast') {
          throw new Error("wasm backend does not support dataFormat:'" +
              (convInfo.dataFormat + "'. Please use 'channelsLast'."));
      }
      if (convInfo.dilationWidth !== 1 || convInfo.dilationHeight !== 1) {
          throw new Error("was backend only supports average pooling with dilation = [1, 1], " +
              ("got [" + convInfo.dilationHeight + ", " + convInfo.dilationWidth + "]."));
      }
      var out = backend.makeOutput(convInfo.outShape, 'float32');
      var outId = backend.dataIdMap.get(out.dataId).id;
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
  var wasmBatchMatMul;
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
      var inputs = args.inputs, backend = args.backend, attrs = args.attrs;
      var a = inputs.a, b = inputs.b;
      if (a.dtype !== 'float32' || b.dtype !== 'float32') {
          throw new Error("BatchMatMul for non non-float32 tensors not yet supported.");
      }
      var transposeA = attrs.transposeA, transposeB = attrs.transposeB;
      var aId = backend.dataIdMap.get(a.dataId).id;
      var bId = backend.dataIdMap.get(b.dataId).id;
      var leftDim = transposeA ? a.shape[2] : a.shape[1];
      var rightDim = transposeB ? b.shape[1] : b.shape[2];
      var batchDim = a.shape[0];
      var out = backend.makeOutput([batchDim, leftDim, rightDim], a.dtype);
      var outId = backend.dataIdMap.get(out.dataId).id;
      var aShapeBytes = new Uint8Array(new Int32Array(a.shape).buffer);
      var bShapeBytes = new Uint8Array(new Int32Array(b.shape).buffer);
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
      var x = args.inputs.x, dtype = args.attrs.dtype, backend = args.backend;
      var out = backend.makeOutput(x.shape, dtype);
      var inVals = backend.typedArrayFromHeap(x);
      var outVals = backend.typedArrayFromHeap(out);
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
  var wasmClip;
  function setup$4(backend) {
      wasmClip = backend.wasm.cwrap('ClipByValue', null /* void */, [
          'number',
          'number',
          'number',
          'number' // out_id
      ]);
  }
  function clip(args) {
      var inputs = args.inputs, backend = args.backend, attrs = args.attrs;
      var x = inputs.x;
      var min = attrs.min, max = attrs.max;
      var xId = backend.dataIdMap.get(x.dataId).id;
      var out = backend.makeOutput(x.shape, 'float32');
      var outId = backend.dataIdMap.get(out.dataId).id;
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
      var inputs = args.inputs, backend = args.backend, axis = args.attrs.axis;
      var outShape = tfjsCore.backend_util.computeOutShape(inputs.map(function (t) { return t.shape; }), axis);
      var out = backend.makeOutput(outShape, inputs[0].dtype);
      var batchDim = tfjsCore.util.sizeFromShape(inputs[0].shape.slice(0, axis));
      var sumInnerDims = 0;
      var innerDims = inputs.map(function (input) {
          var innerDim = tfjsCore.util.sizeFromShape(input.shape.slice(axis));
          sumInnerDims += innerDim;
          return innerDim;
      });
      var inVals = inputs.map(function (input) { return backend.typedArrayFromHeap(input); });
      var outVals = backend.typedArrayFromHeap(out);
      for (var b = 0; b < batchDim; b++) {
          var outOffset = b * sumInnerDims;
          for (var i = 0; i < inVals.length; i++) {
              var innerDim = innerDims[i];
              var inOffset = b * innerDim;
              var vals = inVals[i].subarray(inOffset, inOffset + innerDim);
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
  var wasmConv2d;
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
      var inputs = args.inputs, attrs = args.attrs, backend = args.backend;
      var convInfo = attrs;
      var x = inputs.x, filter = inputs.filter;
      var xId = backend.dataIdMap.get(x.dataId).id;
      var filterId = backend.dataIdMap.get(filter.dataId).id;
      var filterHeight = convInfo.filterHeight;
      var filterWidth = convInfo.filterWidth;
      var padTop = convInfo.padInfo.top;
      var padRight = convInfo.padInfo.right;
      var padBottom = convInfo.padInfo.bottom;
      var padLeft = convInfo.padInfo.left;
      var dilationHeight = convInfo.dilationHeight;
      var dilationWidth = convInfo.dilationWidth;
      var strideHeight = convInfo.strideHeight;
      var strideWidth = convInfo.strideWidth;
      var inputChannels = convInfo.inChannels;
      var outputChannels = convInfo.outChannels;
      var isSamePad = convInfo.padInfo.type === 'SAME' ? 1 : 0;
      if (convInfo.dataFormat !== 'channelsLast') {
          throw new Error("wasm backend Conv2D does not support dataFormat:'" +
              (convInfo.dataFormat + "'. Please use 'channelsLast'."));
      }
      var out = backend.makeOutput(convInfo.outShape, 'float32');
      var outId = backend.dataIdMap.get(out.dataId).id;
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
  var wasmCropAndResize;
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
      var backend = args.backend, inputs = args.inputs, attrs = args.attrs;
      var method = attrs.method, extrapolationValue = attrs.extrapolationValue, cropSize = attrs.cropSize;
      var images = inputs.images, boxes = inputs.boxes, boxInd = inputs.boxInd;
      var numBoxes = boxes.shape[0];
      var _a = cropSize, cropHeight = _a[0], cropWidth = _a[1];
      var outShape = [numBoxes, cropHeight, cropWidth, images.shape[3]];
      var imagesData = backend.dataIdMap.get(images.dataId);
      var castedData;
      if (images.dtype !== 'float32') {
          castedData =
              cast({ backend: backend, inputs: { x: images }, attrs: { dtype: 'float32' } });
          imagesData = backend.dataIdMap.get(castedData.dataId);
      }
      var imagesId = imagesData.id;
      var boxesId = backend.dataIdMap.get(boxes.dataId).id;
      var boxIndId = backend.dataIdMap.get(boxInd.dataId).id;
      var out = backend.makeOutput(outShape, 'float32');
      var outId = backend.dataIdMap.get(out.dataId).id;
      var imagesShapeBytes = new Uint8Array(new Int32Array(images.shape).buffer);
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
  var wasmDepthwiseConv2d;
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
      var inputs = args.inputs, attrs = args.attrs, backend = args.backend;
      var convInfo = attrs;
      var x = inputs.x, filter = inputs.filter;
      var xId = backend.dataIdMap.get(x.dataId).id;
      var filterId = backend.dataIdMap.get(filter.dataId).id;
      var filterHeight = convInfo.filterHeight;
      var filterWidth = convInfo.filterWidth;
      var padTop = convInfo.padInfo.top;
      var padRight = convInfo.padInfo.right;
      var padBottom = convInfo.padInfo.bottom;
      var padLeft = convInfo.padInfo.left;
      var dilationHeight = convInfo.dilationHeight;
      var dilationWidth = convInfo.dilationWidth;
      var strideHeight = convInfo.strideHeight;
      var strideWidth = convInfo.strideWidth;
      var inputChannels = convInfo.inChannels;
      var outputChannels = convInfo.outChannels;
      var isSamePad = convInfo.padInfo.type === 'SAME' ? 1 : 0;
      if (convInfo.dataFormat !== 'channelsLast') {
          throw new Error("wasm backend DepthwiseConv2dNative does not support dataFormat:'" +
              (convInfo.dataFormat + "'. Please use 'channelsLast'."));
      }
      var out = backend.makeOutput(convInfo.outShape, 'float32');
      var outId = backend.dataIdMap.get(out.dataId).id;
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
  var supportsFullBroadcast$1 = false;
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
  var supportsFullBroadcast$2 = false;
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
  var wasmBatchNorm;
  function setup$8(backend) {
      wasmBatchNorm = backend.wasm.cwrap('FusedBatchNorm', null /* void */, ['number', 'number', 'number', 'number', 'number', 'number', 'number']);
  }
  function fusedBatchNorm(args) {
      var backend = args.backend, inputs = args.inputs, attrs = args.attrs;
      var varianceEpsilon = attrs.varianceEpsilon;
      var x = inputs.x, mean = inputs.mean, variance = inputs.variance, offset = inputs.offset, scale = inputs.scale;
      var xId = backend.dataIdMap.get(x.dataId).id;
      var meanId = backend.dataIdMap.get(mean.dataId).id;
      var varianceId = backend.dataIdMap.get(variance.dataId).id;
      var offsetId = offset != null ? backend.dataIdMap.get(offset.dataId).id : 0;
      var scaleId = scale != null ? backend.dataIdMap.get(scale.dataId).id : 0;
      var out = backend.makeOutput(x.shape, x.dtype);
      // Short-circuit zero-sized tensors.
      if (tfjsCore.util.sizeFromShape(x.shape) === 0) {
          return out;
      }
      var outId = backend.dataIdMap.get(out.dataId).id;
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
  var wasmFusedConv2d;
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
      var inputs = args.inputs, attrs = args.attrs, backend = args.backend;
      var convInfo = attrs.convInfo, activation = attrs.activation;
      var fusedActivation = FusableActivation[activation];
      if (fusedActivation == null) {
          throw new Error(activation + " activation not yet supported for FusedConv2D " +
              "in the wasm backend.");
      }
      var x = inputs.x, filter = inputs.filter, bias = inputs.bias, preluActivationWeights = inputs.preluActivationWeights;
      var xId = backend.dataIdMap.get(x.dataId).id;
      var filterId = backend.dataIdMap.get(filter.dataId).id;
      var outputChannels = convInfo.outChannels;
      var biasId = 0;
      if (bias != null) {
          var biasData = backend.dataIdMap.get(bias.dataId);
          if (biasData.shape.length !== 1) {
              throw new Error("FusedConv2D only supports rank-1 bias but got " +
                  ("rank " + biasData.shape.length + "."));
          }
          if (biasData.shape[0] !== outputChannels) {
              throw new Error("FusedConv2D bias shape (" + biasData.shape + ") does not " +
                  ("match the number of output channels (" + outputChannels + ")"));
          }
          biasId = biasData.id;
      }
      var filterHeight = convInfo.filterHeight;
      var filterWidth = convInfo.filterWidth;
      var padTop = convInfo.padInfo.top;
      var padRight = convInfo.padInfo.right;
      var padBottom = convInfo.padInfo.bottom;
      var padLeft = convInfo.padInfo.left;
      var dilationHeight = convInfo.dilationHeight;
      var dilationWidth = convInfo.dilationWidth;
      var strideHeight = convInfo.strideHeight;
      var strideWidth = convInfo.strideWidth;
      var inputChannels = convInfo.inChannels;
      var isSamePad = convInfo.padInfo.type === 'SAME' ? 1 : 0;
      var batchSize = convInfo.batchSize;
      var inHeight = convInfo.inHeight;
      var inWidth = convInfo.inWidth;
      if (convInfo.dataFormat !== 'channelsLast') {
          throw new Error("wasm backend FusedConv2D does not support dataFormat:'" +
              (convInfo.dataFormat + "'. Please use 'channelsLast'."));
      }
      var out = backend.makeOutput(convInfo.outShape, 'float32');
      var outId = backend.dataIdMap.get(out.dataId).id;
      var preluActivationWeightsId = preluActivationWeights == null ?
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
  var wasmFusedDepthwiseConv2d;
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
      var inputs = args.inputs, attrs = args.attrs, backend = args.backend;
      var convInfo = attrs.convInfo, activation = attrs.activation;
      var fusedActivation = FusableActivation[activation];
      if (fusedActivation == null) {
          throw new Error(activation + " activation not yet supported for FusedDepthwiseConv2D " +
              "in the wasm backend.");
      }
      var x = inputs.x, filter = inputs.filter, bias = inputs.bias, preluActivationWeights = inputs.preluActivationWeights;
      var xId = backend.dataIdMap.get(x.dataId).id;
      var filterId = backend.dataIdMap.get(filter.dataId).id;
      var outputChannels = convInfo.outChannels;
      var biasId = 0;
      if (bias != null) {
          var biasData = backend.dataIdMap.get(bias.dataId);
          if (biasData.shape.length !== 1) {
              throw new Error("FusedDepthwiseConv2D only supports rank-1 bias but got " +
                  ("rank " + biasData.shape.length + "."));
          }
          if (biasData.shape[0] !== outputChannels) {
              throw new Error("FusedDepthwiseConv2D bias shape (" + biasData.shape + ") does not " +
                  ("match the number of output channels (" + outputChannels + ")"));
          }
          biasId = biasData.id;
      }
      var filterHeight = convInfo.filterHeight;
      var filterWidth = convInfo.filterWidth;
      var padTop = convInfo.padInfo.top;
      var padRight = convInfo.padInfo.right;
      var padBottom = convInfo.padInfo.bottom;
      var padLeft = convInfo.padInfo.left;
      var dilationHeight = convInfo.dilationHeight;
      var dilationWidth = convInfo.dilationWidth;
      var strideHeight = convInfo.strideHeight;
      var strideWidth = convInfo.strideWidth;
      var inputChannels = convInfo.inChannels;
      var isSamePad = convInfo.padInfo.type === 'SAME' ? 1 : 0;
      var batchSize = convInfo.batchSize;
      var inHeight = convInfo.inHeight;
      var inWidth = convInfo.inWidth;
      if (convInfo.dataFormat !== 'channelsLast') {
          throw new Error("wasm backend FusedDepthwiseConv2D does not support dataFormat:'" +
              (convInfo.dataFormat + "'. Please use 'channelsLast'."));
      }
      var out = backend.makeOutput(convInfo.outShape, 'float32');
      var outId = backend.dataIdMap.get(out.dataId).id;
      var preluActivationWeightsId = preluActivationWeights == null ?
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
  var wasmGather;
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
      var backend = args.backend, inputs = args.inputs, attrs = args.attrs;
      var x = inputs.x, indices = inputs.indices;
      var axis = attrs.axis;
      var newShape = x.shape.slice();
      newShape[axis] = tfjsCore.util.sizeFromShape(indices.shape);
      var stridesSize = x.shape.length - 1;
      var out = backend.makeOutput(newShape, x.dtype);
      if (tfjsCore.util.sizeFromShape(x.shape) === 0) {
          return out;
      }
      var xData = backend.dataIdMap.get(x.dataId);
      var xId = xData.id;
      var indicesData = backend.dataIdMap.get(indices.dataId);
      var indicesId = indicesData.id;
      var outId = backend.dataIdMap.get(out.dataId).id;
      var xStridesBytes = new Uint8Array(new Int32Array(tfjsCore.util.computeStrides(x.shape)).buffer);
      var outStridesBytes = new Uint8Array(new Int32Array(tfjsCore.util.computeStrides(newShape)).buffer);
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
  var wasmGatherNd;
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
      var backend = args.backend, inputs = args.inputs;
      var x = inputs.x, indices = inputs.indices;
      var _a = tfjsCore.gather_util.prepareAndValidate(x, indices), resultShape = _a[0], numSlices = _a[1], sliceSize = _a[2], strides = _a[3];
      var out = backend.makeOutput(resultShape, x.dtype);
      if (numSlices === 0) {
          return out;
      }
      var indicesShape = indices.shape;
      var sliceRank = indicesShape[indicesShape.length - 1];
      var xData = backend.dataIdMap.get(x.dataId);
      var xId = xData.id;
      var indicesData = backend.dataIdMap.get(indices.dataId);
      var indicesId = indicesData.id;
      var stridesBytes = new Uint8Array(new Int32Array(strides).buffer);
      var outId = backend.dataIdMap.get(out.dataId).id;
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
  var supportsFullBroadcast$3 = false;
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
  var supportsFullBroadcast$4 = false;
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
  var supportsFullBroadcast$5 = false;
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
  var supportsFullBroadcast$6 = false;
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
  var supportsFullBroadcast$7 = false;
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
  var wasmMax;
  function setup$d(backend) {
      wasmMax =
          backend.wasm.cwrap('Max', null /*void*/, ['number, number, number']);
  }
  function max(args) {
      var backend = args.backend, inputs = args.inputs, attrs = args.attrs;
      var axes = attrs.axes;
      var x = inputs.x;
      var xId = backend.dataIdMap.get(x.dataId).id;
      tfjsCore.backend_util.assertAxesAreInnerMostDims('max', axes, x.shape.length);
      var _a = tfjsCore.backend_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
      var reduceSize = tfjsCore.util.sizeFromShape(reduceShape);
      var out = backend.makeOutput(outShape, x.dtype);
      if (tfjsCore.util.sizeFromShape(x.shape) === 0) {
          return out;
      }
      var outId = backend.dataIdMap.get(out.dataId).id;
      wasmMax(xId, reduceSize, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'Max',
      backendName: 'wasm',
      setupFunc: setup$d,
      kernelFunc: max
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
  var supportsFullBroadcast$8 = false;
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
  var wasmMaxPool;
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
      var inputs = args.inputs, attrs = args.attrs, backend = args.backend;
      var convInfo = attrs;
      var x = inputs.x;
      var xId = backend.dataIdMap.get(x.dataId).id;
      var filterHeight = convInfo.filterHeight;
      var filterWidth = convInfo.filterWidth;
      var padTop = convInfo.padInfo.top;
      var padRight = convInfo.padInfo.right;
      var padBottom = convInfo.padInfo.bottom;
      var padLeft = convInfo.padInfo.left;
      var dilationHeight = convInfo.dilationHeight;
      var dilationWidth = convInfo.dilationWidth;
      var strideHeight = convInfo.strideHeight;
      var strideWidth = convInfo.strideWidth;
      var inputChannels = convInfo.inChannels;
      var outputChannels = convInfo.outChannels;
      if (convInfo.dataFormat !== 'channelsLast') {
          throw new Error("wasm backend does not support dataFormat:'" +
              (convInfo.dataFormat + "'. Please use 'channelsLast'."));
      }
      var out = backend.makeOutput(convInfo.outShape, 'float32');
      var outId = backend.dataIdMap.get(out.dataId).id;
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
  var wasmMin;
  function setup$f(backend) {
      wasmMin =
          backend.wasm.cwrap('Min', null /*void*/, ['number, number, number']);
  }
  function min(args) {
      var backend = args.backend, inputs = args.inputs, attrs = args.attrs;
      var axes = attrs.axes;
      var x = inputs.x;
      var xId = backend.dataIdMap.get(x.dataId).id;
      tfjsCore.backend_util.assertAxesAreInnerMostDims('min', axes, x.shape.length);
      var _a = tfjsCore.backend_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
      var reduceSize = tfjsCore.util.sizeFromShape(reduceShape);
      var out = backend.makeOutput(outShape, x.dtype);
      if (tfjsCore.util.sizeFromShape(x.shape) === 0) {
          return out;
      }
      var outId = backend.dataIdMap.get(out.dataId).id;
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
  var supportsFullBroadcast$9 = false;
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
  var supportsFullBroadcast$a = true;
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
      var result = new Int32Array(backend.wasm.HEAPU8.buffer, resOffset, 3);
      var pSelectedIndices = result[0];
      var selectedSize = result[1];
      var pSelectedScores = result[2];
      // Since the result was allocated on the heap, we have to delete it.
      backend.wasm._free(resOffset);
      return { pSelectedIndices: pSelectedIndices, selectedSize: selectedSize, pSelectedScores: pSelectedScores };
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
  var wasmFunc$2;
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
      var backend = args.backend, inputs = args.inputs, attrs = args.attrs;
      var iouThreshold = attrs.iouThreshold, maxOutputSize = attrs.maxOutputSize, scoreThreshold = attrs.scoreThreshold;
      var boxes = inputs.boxes, scores = inputs.scores;
      var boxesId = backend.dataIdMap.get(boxes.dataId).id;
      var scoresId = backend.dataIdMap.get(scores.dataId).id;
      var resOffset = wasmFunc$2(boxesId, scoresId, maxOutputSize, iouThreshold, scoreThreshold);
      var _a = parseResultStruct(backend, resOffset), pSelectedIndices = _a.pSelectedIndices, selectedSize = _a.selectedSize, pSelectedScores = _a.pSelectedScores;
      // Since we are not using scores for V3, we have to delete it from the heap.
      backend.wasm._free(pSelectedScores);
      var selectedIndicesTensor = backend.makeOutput([selectedSize], 'int32', pSelectedIndices);
      return selectedIndicesTensor;
  }
  tfjsCore.registerKernel({
      kernelName: 'NonMaxSuppressionV3',
      backendName: 'wasm',
      setupFunc: setup$g,
      kernelFunc: kernelFunc,
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
  var wasmFunc$3;
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
      var backend = args.backend, inputs = args.inputs, attrs = args.attrs;
      var iouThreshold = attrs.iouThreshold, maxOutputSize = attrs.maxOutputSize, scoreThreshold = attrs.scoreThreshold, softNmsSigma = attrs.softNmsSigma;
      var boxes = inputs.boxes, scores = inputs.scores;
      var boxesId = backend.dataIdMap.get(boxes.dataId).id;
      var scoresId = backend.dataIdMap.get(scores.dataId).id;
      var resOffset = wasmFunc$3(boxesId, scoresId, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma);
      var _a = parseResultStruct(backend, resOffset), pSelectedIndices = _a.pSelectedIndices, selectedSize = _a.selectedSize, pSelectedScores = _a.pSelectedScores;
      var selectedIndicesTensor = backend.makeOutput([selectedSize], 'int32', pSelectedIndices);
      var selectedScoresTensor = backend.makeOutput([selectedSize], 'float32', pSelectedScores);
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
  var supportsFullBroadcast$b = false;
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
      var x = args.inputs.x, backend = args.backend;
      var out = backend.makeOutput(x.shape, x.dtype);
      var outVals = backend.typedArrayFromHeap(out);
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
  var wasmPadV2;
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
      var x = args.inputs.x, backend = args.backend, _a = args.attrs, paddings = _a.paddings, constantValue = _a.constantValue;
      var outShape = paddings.map(function (p, i) { return p[0] /* beforePad */ + x.shape[i] + p[1]; } /* afterPad */);
      var xId = backend.dataIdMap.get(x.dataId).id;
      var out = backend.makeOutput(outShape, x.dtype);
      var outId = backend.dataIdMap.get(out.dataId).id;
      var xShapeBytes = new Uint8Array(new Int32Array(x.shape).buffer);
      var paddingsFlat = [].concat.apply([], paddings);
      var paddingsBytes = new Uint8Array(new Int32Array(paddingsFlat).buffer);
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
  var supportsFullBroadcast$c = false;
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
  var wasmPrelu;
  function setup$j(backend) {
      wasmPrelu = backend.wasm.cwrap('Prelu', null /* void */, [
          'number',
          'number',
          'number' // out_id
      ]);
  }
  function prelu(args) {
      var inputs = args.inputs, backend = args.backend;
      var x = inputs.x, alpha = inputs.alpha;
      var xId = backend.dataIdMap.get(x.dataId).id;
      var weightsId = backend.dataIdMap.get(alpha.dataId).id;
      var out = backend.makeOutput(x.shape, 'float32');
      var outId = backend.dataIdMap.get(out.dataId).id;
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
      var x = args.inputs.x, shape = args.attrs.shape;
      return { dataId: x.dataId, shape: shape, dtype: x.dtype };
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
  var wasmResizeBilinear;
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
      var backend = args.backend, inputs = args.inputs, attrs = args.attrs;
      var x = inputs.x;
      var alignCorners = attrs.alignCorners, newHeight = attrs.newHeight, newWidth = attrs.newWidth;
      var _a = x.shape, batch = _a[0], oldHeight = _a[1], oldWidth = _a[2], numChannels = _a[3];
      var outShape = [batch, newHeight, newWidth, numChannels];
      var xData = backend.dataIdMap.get(x.dataId);
      var castedData;
      if (xData.dtype !== 'float32') {
          castedData = cast({ backend: backend, inputs: { x: x }, attrs: { dtype: 'float32' } });
          xData = backend.dataIdMap.get(castedData.dataId);
      }
      var xId = xData.id;
      var out = backend.makeOutput(outShape, 'float32');
      if (tfjsCore.util.sizeFromShape(x.shape) === 0) {
          return out;
      }
      var outId = backend.dataIdMap.get(out.dataId).id;
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
  var wasmScatterNd;
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
      var backend = args.backend, inputs = args.inputs, attrs = args.attrs;
      var indices = inputs.indices, updates = inputs.updates;
      var shape = attrs.shape;
      var out = backend.makeOutput(shape, updates.dtype);
      if (tfjsCore.util.sizeFromShape(shape) === 0) {
          return out;
      }
      var _a = tfjsCore.scatter_util.calculateShapes(updates, indices, shape), sliceRank = _a.sliceRank, numUpdates = _a.numUpdates, sliceSize = _a.sliceSize, strides = _a.strides, outputSize = _a.outputSize;
      var indicesData = backend.dataIdMap.get(indices.dataId);
      var indicesId = indicesData.id;
      var updatesData = backend.dataIdMap.get(updates.dataId);
      var updatesId = updatesData.id;
      var stridesBytes = new Uint8Array(new Int32Array(strides).buffer);
      var outId = backend.dataIdMap.get(out.dataId).id;
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
  var wasmFunc$4;
  function setup$m(backend) {
      wasmFunc$4 =
          backend.wasm.cwrap('Sigmoid', null /* void */, ['number', 'number']);
  }
  function sigmoid(args) {
      var backend = args.backend, x = args.inputs.x;
      var xId = backend.dataIdMap.get(x.dataId).id;
      var out = backend.makeOutput(x.shape, x.dtype);
      var outId = backend.dataIdMap.get(out.dataId).id;
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
      var x = args.inputs.x, _a = args.attrs, begin = _a.begin, size = _a.size, backend = args.backend;
      var isContinous = tfjsCore.slice_util.isSliceContinous(x.shape, begin, size);
      var xVals = backend.typedArrayFromHeap(x);
      var out = backend.makeOutput(size, x.dtype);
      var outVals = backend.typedArrayFromHeap(out);
      var xStrides = tfjsCore.util.computeStrides(x.shape);
      if (isContinous) {
          var flatOffset = tfjsCore.slice_util.computeFlatOffset(begin, xStrides);
          outVals.set(xVals.subarray(flatOffset, flatOffset + tfjsCore.util.sizeFromShape(size)));
          return out;
      }
      var rank = x.shape.length;
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
      var outOffset = 0;
      var beginI = begin[0];
      var beginJ = begin[1];
      var endI = beginI + size[0];
      for (var i = beginI; i < endI; i++) {
          var xOffset = i * xStride + beginJ;
          outVals.set(xVals.subarray(xOffset, xOffset + size[1]), outOffset);
          outOffset += size[1];
      }
  }
  function slice3d(xVals, xStride1, xStride2, outVals, begin, size) {
      var outOffset = 0;
      var beginI = begin[0];
      var beginJ = begin[1];
      var beginK = begin[2];
      var endI = beginI + size[0];
      var endJ = beginJ + size[1];
      for (var i = beginI; i < endI; i++) {
          for (var j = beginJ; j < endJ; j++) {
              var xOffset = i * xStride1 + j * xStride2 + beginK;
              outVals.set(xVals.subarray(xOffset, xOffset + size[2]), outOffset);
              outOffset += size[2];
          }
      }
  }
  function slice4d(xVals, xStride1, xStride2, xStride3, outVals, begin, size) {
      var outOffset = 0;
      var beginI = begin[0];
      var beginJ = begin[1];
      var beginK = begin[2];
      var endI = beginI + size[0];
      var endJ = beginJ + size[1];
      var endK = beginK + size[2];
      var beginL = begin[3];
      for (var i = beginI; i < endI; i++) {
          for (var j = beginJ; j < endJ; j++) {
              for (var k = beginK; k < endK; k++) {
                  var xOffset = i * xStride1 + j * xStride2 + k * xStride3 + beginL;
                  outVals.set(xVals.subarray(xOffset, xOffset + size[3]), outOffset);
                  outOffset += size[3];
              }
          }
      }
  }
  function genericSliceSlow(xVals, xInfo, outVals, begin, size) {
      var outBuf = tfjsCore.buffer(size, xInfo.dtype, outVals);
      var xBuf = tfjsCore.buffer(xInfo.shape, xInfo.dtype, xVals);
      for (var i = 0; i < outBuf.size; ++i) {
          var loc = outBuf.indexToLoc(i);
          var xLoc = loc.map(function (idx, j) { return idx + begin[j]; });
          outVals[i] = xBuf.get.apply(xBuf, xLoc);
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
  var wasmFunc$5;
  function setup$n(backend) {
      wasmFunc$5 = backend.wasm.cwrap('Softmax', null /* void */, [
          'number',
          'number',
          'number',
          'number' // batch
      ]);
  }
  function softmax(args) {
      var backend = args.backend, logits = args.inputs.logits, dim = args.attrs.dim;
      var xId = backend.dataIdMap.get(logits.dataId).id;
      var out = backend.makeOutput(logits.shape, logits.dtype);
      var outId = backend.dataIdMap.get(out.dataId).id;
      var channels = logits.shape[dim];
      var batch = tfjsCore.util.sizeFromShape(logits.shape) / channels;
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
  var supportsFullBroadcast$d = true;
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
  var wasmSum;
  function setup$o(backend) {
      wasmSum =
          backend.wasm.cwrap('Sum', null /*void*/, ['number, number, number']);
  }
  function sum(args) {
      var backend = args.backend, inputs = args.inputs, attrs = args.attrs;
      var axes = attrs.axes;
      var x = inputs.x;
      var xId = backend.dataIdMap.get(x.dataId).id;
      tfjsCore.backend_util.assertAxesAreInnerMostDims('sum', axes, x.shape.length);
      var _a = tfjsCore.backend_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
      var reduceSize = tfjsCore.util.sizeFromShape(reduceShape);
      var out = backend.makeOutput(outShape, x.dtype);
      if (tfjsCore.util.sizeFromShape(x.shape) === 0) {
          return out;
      }
      var outId = backend.dataIdMap.get(out.dataId).id;
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
  var wasmTile;
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
      var inputs = args.inputs, backend = args.backend, attrs = args.attrs;
      var x = inputs.x;
      var xId = backend.dataIdMap.get(x.dataId).id;
      var reps = attrs.reps;
      var newShape = new Array(x.shape.length);
      for (var i = 0; i < newShape.length; i++) {
          newShape[i] = x.shape[i] * reps[i];
      }
      var xShapeBytes = new Uint8Array(new Int32Array(x.shape).buffer);
      var newShapeBytes = new Uint8Array(new Int32Array(newShape).buffer);
      var out = backend.makeOutput(newShape, x.dtype);
      var outId = backend.dataIdMap.get(out.dataId).id;
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
  var wasmTranspose;
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
      var inputs = args.inputs, backend = args.backend, attrs = args.attrs;
      // Reduce any dimensions with size one. Lower-rank transpose kernel performs
      // better due to simpler memory access pattern.
      var _a = removeOneSizeDims(inputs.x.shape, attrs.perm), reducedShape = _a[0], perm = _a[1];
      var x = {
          dataId: inputs.x.dataId,
          shape: reducedShape,
          dtype: inputs.x.dtype
      };
      var permIsNoOp = true;
      for (var i = 0; i < perm.length; i++) {
          if (perm[i] !== i) {
              permIsNoOp = false;
          }
      }
      var outShape = computeOutShape(inputs.x.shape, attrs.perm);
      if (permIsNoOp) {
          return { dataId: x.dataId, shape: outShape, dtype: x.dtype };
      }
      var out = backend.makeOutput(outShape, x.dtype);
      var xId = backend.dataIdMap.get(x.dataId).id;
      var outId = backend.dataIdMap.get(out.dataId).id;
      var permBytes = new Uint8Array(new Int32Array(perm).buffer);
      var xShapeBytes = new Uint8Array(new Int32Array(x.shape).buffer);
      wasmTranspose(xId, xShapeBytes, x.shape.length, CppDType[x.dtype], outId, permBytes, perm.length);
      return out;
  }
  function computeOutShape(inShape, perm) {
      var outShape = new Array(inShape.length);
      for (var i = 0; i < outShape.length; i++) {
          outShape[i] = inShape[perm[i]];
      }
      return outShape;
  }
  function removeOneSizeDims(shape, perm) {
      var newShape = [];
      var newPerm = [];
      for (var i = 0; i < shape.length; ++i) {
          if (shape[i] !== 1) {
              newShape.push(shape[i]);
          }
          if (shape[perm[i]] !== 1) {
              newPerm.push(perm[i]);
          }
      }
      for (var i = 0; i < newPerm.length; ++i) {
          var minValIdx = -1;
          for (var j = 0; j < newPerm.length; ++j) {
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
      var x = args.inputs.x, backend = args.backend, axis = args.attrs.axis;
      var numOutputs = x.shape[axis];
      var rank = x.shape.length;
      var outShape = new Array(rank - 1);
      var outIndex = 0;
      for (var i = 0; i < rank; i++) {
          if (i !== axis) {
              outShape[outIndex++] = x.shape[i];
          }
      }
      var outs = new Array(numOutputs);
      var begin = new Array(rank).fill(0);
      var size = x.shape.slice();
      size[axis] = 1;
      for (var i = 0; i < outs.length; i++) {
          begin[axis] = i;
          outs[i] = slice({ inputs: { x: x }, attrs: { begin: begin, size: size }, backend: backend });
      }
      return outs.map(function (_a) {
          var dataId = _a.dataId, dtype = _a.dtype;
          return ({ dataId: dataId, dtype: dtype, shape: outShape });
      });
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
      var x = args.inputs.x, backend = args.backend;
      var out = backend.makeOutput(x.shape, x.dtype);
      var outVals = backend.typedArrayFromHeap(out);
      outVals.fill(0);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'ZerosLike',
      backendName: 'wasm',
      kernelFunc: zerosLike,
  });

  /*! *****************************************************************************
  Copyright (c) Microsoft Corporation. All rights reserved.
  Licensed under the Apache License, Version 2.0 (the "License"); you may not use
  this file except in compliance with the License. You may obtain a copy of the
  License at http://www.apache.org/licenses/LICENSE-2.0

  THIS CODE IS PROVIDED ON AN *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
  WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
  MERCHANTABLITY OR NON-INFRINGEMENT.

  See the Apache Version 2.0 License for specific language governing permissions
  and limitations under the License.
  ***************************************************************************** */
  /* global Reflect, Promise */

  var extendStatics = function(d, b) {
      extendStatics = Object.setPrototypeOf ||
          ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
          function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
      return extendStatics(d, b);
  };

  function __extends(d, b) {
      extendStatics(d, b);
      function __() { this.constructor = d; }
      d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
  }

  function __awaiter(thisArg, _arguments, P, generator) {
      return new (P || (P = Promise))(function (resolve, reject) {
          function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
          function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
          function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
          step((generator = generator.apply(thisArg, _arguments || [])).next());
      });
  }

  function __generator(thisArg, body) {
      var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
      return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
      function verb(n) { return function (v) { return step([n, v]); }; }
      function step(op) {
          if (f) throw new TypeError("Generator is already executing.");
          while (_) try {
              if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
              if (y = 0, t) op = [op[0] & 2, t.value];
              switch (op[0]) {
                  case 0: case 1: t = op; break;
                  case 4: _.label++; return { value: op[1], done: false };
                  case 5: _.label++; y = op[1]; op = [0]; continue;
                  case 7: op = _.ops.pop(); _.trys.pop(); continue;
                  default:
                      if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                      if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                      if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                      if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                      if (t[2]) _.ops.pop();
                      _.trys.pop(); continue;
              }
              op = body.call(thisArg, _);
          } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
          if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
      }
  }

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

  function GROWABLE_HEAP_I8(){if(wasmMemory.buffer!=buffer){updateGlobalBufferAndViews(wasmMemory.buffer);}return HEAP8}function GROWABLE_HEAP_U8(){if(wasmMemory.buffer!=buffer){updateGlobalBufferAndViews(wasmMemory.buffer);}return HEAPU8}function GROWABLE_HEAP_I32(){if(wasmMemory.buffer!=buffer){updateGlobalBufferAndViews(wasmMemory.buffer);}return HEAP32}function GROWABLE_HEAP_U32(){if(wasmMemory.buffer!=buffer){updateGlobalBufferAndViews(wasmMemory.buffer);}return HEAPU32}function GROWABLE_HEAP_F64(){if(wasmMemory.buffer!=buffer){updateGlobalBufferAndViews(wasmMemory.buffer);}return HEAPF64}var Module=typeof WasmBackendModule!=="undefined"?WasmBackendModule:{};var moduleOverrides={};var key;for(key in Module){if(Module.hasOwnProperty(key)){moduleOverrides[key]=Module[key];}}var arguments_=[];var thisProgram="./this.program";var quit_=function(status,toThrow){throw toThrow};var ENVIRONMENT_IS_WEB=false;var ENVIRONMENT_IS_WORKER=false;var ENVIRONMENT_IS_NODE=false;var ENVIRONMENT_IS_SHELL=false;ENVIRONMENT_IS_WEB=typeof window==="object";ENVIRONMENT_IS_WORKER=typeof importScripts==="function";ENVIRONMENT_IS_NODE=typeof process==="object"&&typeof process.versions==="object"&&typeof process.versions.node==="string";ENVIRONMENT_IS_SHELL=!ENVIRONMENT_IS_WEB&&!ENVIRONMENT_IS_NODE&&!ENVIRONMENT_IS_WORKER;if(Module["ENVIRONMENT"]){throw new Error("Module.ENVIRONMENT has been deprecated. To force the environment, use the ENVIRONMENT compile-time option (for example, -s ENVIRONMENT=web or -s ENVIRONMENT=node)")}var ENVIRONMENT_IS_PTHREAD=Module["ENVIRONMENT_IS_PTHREAD"]||false;if(ENVIRONMENT_IS_PTHREAD){buffer=Module["buffer"];DYNAMIC_BASE=Module["DYNAMIC_BASE"];DYNAMICTOP_PTR=Module["DYNAMICTOP_PTR"];}var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var read_,readBinary;var nodeFS;var nodePath;if(ENVIRONMENT_IS_NODE){if(ENVIRONMENT_IS_WORKER){scriptDirectory=path.dirname(scriptDirectory)+"/";}else{scriptDirectory=__dirname+"/";}read_=function shell_read(filename,binary){if(!nodeFS)nodeFS=fs;if(!nodePath)nodePath=path;filename=nodePath["normalize"](filename);return nodeFS["readFileSync"](filename,binary?null:"utf8")};readBinary=function readBinary(filename){var ret=read_(filename,true);if(!ret.buffer){ret=new Uint8Array(ret);}assert(ret.buffer);return ret};if(process["argv"].length>1){thisProgram=process["argv"][1].replace(/\\/g,"/");}arguments_=process["argv"].slice(2);process["on"]("uncaughtException",function(ex){if(!(ex instanceof ExitStatus)){throw ex}});process["on"]("unhandledRejection",abort);quit_=function(status){process["exit"](status);};Module["inspect"]=function(){return "[Emscripten Module object]"};var nodeWorkerThreads;try{nodeWorkerThreads=worker_threads;}catch(e){console.error('The "worker_threads" module is not supported in this node.js build - perhaps a newer version is needed?');throw e}Worker=nodeWorkerThreads.Worker;}else if(ENVIRONMENT_IS_SHELL){if(typeof read!="undefined"){read_=function shell_read(f){return read(f)};}readBinary=function readBinary(f){var data;if(typeof readbuffer==="function"){return new Uint8Array(readbuffer(f))}data=read(f,"binary");assert(typeof data==="object");return data};if(typeof scriptArgs!="undefined"){arguments_=scriptArgs;}else if(typeof arguments!="undefined"){arguments_=arguments;}if(typeof quit==="function"){quit_=function(status){quit(status);};}if(typeof print!=="undefined"){if(typeof console==="undefined")console={};console.log=print;console.warn=console.error=typeof printErr!=="undefined"?printErr:print;}}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){if(ENVIRONMENT_IS_WORKER){scriptDirectory=self.location.href;}else if(document.currentScript){scriptDirectory=document.currentScript.src;}if(_scriptDir){scriptDirectory=_scriptDir;}if(scriptDirectory.indexOf("blob:")!==0){scriptDirectory=scriptDirectory.substr(0,scriptDirectory.lastIndexOf("/")+1);}else{scriptDirectory="";}if(ENVIRONMENT_IS_NODE){read_=function shell_read(filename,binary){if(!nodeFS)nodeFS=fs;if(!nodePath)nodePath=path;filename=nodePath["normalize"](filename);return nodeFS["readFileSync"](filename,binary?null:"utf8")};readBinary=function readBinary(filename){var ret=read_(filename,true);if(!ret.buffer){ret=new Uint8Array(ret);}assert(ret.buffer);return ret};}else{read_=function shell_read(url){var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.send(null);return xhr.responseText};if(ENVIRONMENT_IS_WORKER){readBinary=function readBinary(url){var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)};}}}else{throw new Error("environment detection error")}if(ENVIRONMENT_IS_NODE){if(typeof performance==="undefined"){performance=perf_hooks.performance;}}var out=Module["print"]||console.log.bind(console);var err=Module["printErr"]||console.warn.bind(console);for(key in moduleOverrides){if(moduleOverrides.hasOwnProperty(key)){Module[key]=moduleOverrides[key];}}moduleOverrides=null;if(Module["arguments"])arguments_=Module["arguments"];if(!Object.getOwnPropertyDescriptor(Module,"arguments"))Object.defineProperty(Module,"arguments",{configurable:true,get:function(){abort("Module.arguments has been replaced with plain arguments_");}});if(Module["thisProgram"])thisProgram=Module["thisProgram"];if(!Object.getOwnPropertyDescriptor(Module,"thisProgram"))Object.defineProperty(Module,"thisProgram",{configurable:true,get:function(){abort("Module.thisProgram has been replaced with plain thisProgram");}});if(Module["quit"])quit_=Module["quit"];if(!Object.getOwnPropertyDescriptor(Module,"quit"))Object.defineProperty(Module,"quit",{configurable:true,get:function(){abort("Module.quit has been replaced with plain quit_");}});assert(typeof Module["memoryInitializerPrefixURL"]==="undefined","Module.memoryInitializerPrefixURL option was removed, use Module.locateFile instead");assert(typeof Module["pthreadMainPrefixURL"]==="undefined","Module.pthreadMainPrefixURL option was removed, use Module.locateFile instead");assert(typeof Module["cdInitializerPrefixURL"]==="undefined","Module.cdInitializerPrefixURL option was removed, use Module.locateFile instead");assert(typeof Module["filePackagePrefixURL"]==="undefined","Module.filePackagePrefixURL option was removed, use Module.locateFile instead");assert(typeof Module["read"]==="undefined","Module.read option was removed (modify read_ in JS)");assert(typeof Module["readAsync"]==="undefined","Module.readAsync option was removed (modify readAsync in JS)");assert(typeof Module["readBinary"]==="undefined","Module.readBinary option was removed (modify readBinary in JS)");assert(typeof Module["setWindowTitle"]==="undefined","Module.setWindowTitle option was removed (modify setWindowTitle in JS)");assert(typeof Module["TOTAL_MEMORY"]==="undefined","Module.TOTAL_MEMORY has been renamed Module.INITIAL_MEMORY");if(!Object.getOwnPropertyDescriptor(Module,"read"))Object.defineProperty(Module,"read",{configurable:true,get:function(){abort("Module.read has been replaced with plain read_");}});if(!Object.getOwnPropertyDescriptor(Module,"readAsync"))Object.defineProperty(Module,"readAsync",{configurable:true,get:function(){abort("Module.readAsync has been replaced with plain readAsync");}});if(!Object.getOwnPropertyDescriptor(Module,"readBinary"))Object.defineProperty(Module,"readBinary",{configurable:true,get:function(){abort("Module.readBinary has been replaced with plain readBinary");}});if(!Object.getOwnPropertyDescriptor(Module,"setWindowTitle"))Object.defineProperty(Module,"setWindowTitle",{configurable:true,get:function(){abort("Module.setWindowTitle has been replaced with plain setWindowTitle");}});assert(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER||ENVIRONMENT_IS_NODE,"Pthreads do not work in this environment yet (need Web Workers, or an alternative to them)");var stackSave;var stackRestore;var stackAlloc;stackSave=stackRestore=stackAlloc=function(){abort("cannot use the stack before compiled code is ready to run, and has provided stack access");};function warnOnce(text){if(!warnOnce.shown)warnOnce.shown={};if(!warnOnce.shown[text]){warnOnce.shown[text]=1;err(text);}}var wasmBinary;if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];if(!Object.getOwnPropertyDescriptor(Module,"wasmBinary"))Object.defineProperty(Module,"wasmBinary",{configurable:true,get:function(){abort("Module.wasmBinary has been replaced with plain wasmBinary");}});var noExitRuntime;if(Module["noExitRuntime"])noExitRuntime=Module["noExitRuntime"];if(!Object.getOwnPropertyDescriptor(Module,"noExitRuntime"))Object.defineProperty(Module,"noExitRuntime",{configurable:true,get:function(){abort("Module.noExitRuntime has been replaced with plain noExitRuntime");}});if(typeof WebAssembly!=="object"){abort("No WebAssembly support found. Build with -s WASM=0 to target JavaScript instead.");}var wasmMemory;var wasmTable=new WebAssembly.Table({"initial":118,"maximum":118+0,"element":"anyfunc"});var wasmModule;var threadInfoStruct=0;var selfThreadId=0;var ABORT=false;function assert(condition,text){if(!condition){abort("Assertion failed: "+text);}}function getCFunc(ident){var func=Module["_"+ident];assert(func,"Cannot call unknown function "+ident+", make sure it is exported");return func}function ccall(ident,returnType,argTypes,args,opts){var toC={"string":function(str){var ret=0;if(str!==null&&str!==undefined&&str!==0){var len=(str.length<<2)+1;ret=stackAlloc(len);stringToUTF8(str,ret,len);}return ret},"array":function(arr){var ret=stackAlloc(arr.length);writeArrayToMemory(arr,ret);return ret}};function convertReturnValue(ret){if(returnType==="string")return UTF8ToString(ret);if(returnType==="boolean")return Boolean(ret);return ret}var func=getCFunc(ident);var cArgs=[];var stack=0;assert(returnType!=="array",'Return type should not be "array".');if(args){for(var i=0;i<args.length;i++){var converter=toC[argTypes[i]];if(converter){if(stack===0)stack=stackSave();cArgs[i]=converter(args[i]);}else{cArgs[i]=args[i];}}}var ret=func.apply(null,cArgs);ret=convertReturnValue(ret);if(stack!==0)stackRestore(stack);return ret}function cwrap(ident,returnType,argTypes,opts){return function(){return ccall(ident,returnType,argTypes,arguments)}}function UTF8ArrayToString(heap,idx,maxBytesToRead){var endIdx=idx+maxBytesToRead;var str="";while(!(idx>=endIdx)){var u0=heap[idx++];if(!u0)return str;if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=heap[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=heap[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2;}else{if((u0&248)!=240)warnOnce("Invalid UTF-8 leading byte 0x"+u0.toString(16)+" encountered when deserializing a UTF-8 string on the asm.js/wasm heap to a JS string!");u0=(u0&7)<<18|u1<<12|u2<<6|heap[idx++]&63;}if(u0<65536){str+=String.fromCharCode(u0);}else{var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023);}}return str}function UTF8ToString(ptr,maxBytesToRead){return ptr?UTF8ArrayToString(GROWABLE_HEAP_U8(),ptr,maxBytesToRead):""}function stringToUTF8Array(str,heap,outIdx,maxBytesToWrite){if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.charCodeAt(i);if(u>=55296&&u<=57343){var u1=str.charCodeAt(++i);u=65536+((u&1023)<<10)|u1&1023;}if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u;}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63;}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;}else{if(outIdx+3>=endIdx)break;if(u>=2097152)warnOnce("Invalid Unicode code point 0x"+u.toString(16)+" encountered when serializing a JS string to an UTF-8 string on the asm.js/wasm heap! (Valid unicode code points should be in range 0-0x1FFFFF).");heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;}}heap[outIdx]=0;return outIdx-startIdx}function stringToUTF8(str,outPtr,maxBytesToWrite){assert(typeof maxBytesToWrite=="number","stringToUTF8(str, outPtr, maxBytesToWrite) is missing the third parameter that specifies the length of the output buffer!");return stringToUTF8Array(str,GROWABLE_HEAP_U8(),outPtr,maxBytesToWrite)}function lengthBytesUTF8(str){var len=0;for(var i=0;i<str.length;++i){var u=str.charCodeAt(i);if(u>=55296&&u<=57343)u=65536+((u&1023)<<10)|str.charCodeAt(++i)&1023;if(u<=127)++len;else if(u<=2047)len+=2;else if(u<=65535)len+=3;else len+=4;}return len}function writeArrayToMemory(array,buffer){assert(array.length>=0,"writeArrayToMemory array must have a length (should be an array or typed array)");GROWABLE_HEAP_I8().set(array,buffer);}var WASM_PAGE_SIZE=65536;function alignUp(x,multiple){if(x%multiple>0){x+=multiple-x%multiple;}return x}var buffer,HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;function updateGlobalBufferAndViews(buf){buffer=buf;Module["HEAP8"]=HEAP8=new Int8Array(buf);Module["HEAP16"]=HEAP16=new Int16Array(buf);Module["HEAP32"]=HEAP32=new Int32Array(buf);Module["HEAPU8"]=HEAPU8=new Uint8Array(buf);Module["HEAPU16"]=HEAPU16=new Uint16Array(buf);Module["HEAPU32"]=HEAPU32=new Uint32Array(buf);Module["HEAPF32"]=HEAPF32=new Float32Array(buf);Module["HEAPF64"]=HEAPF64=new Float64Array(buf);}var STACK_BASE=5255808,STACKTOP=STACK_BASE,STACK_MAX=12928,DYNAMIC_BASE=5255808,DYNAMICTOP_PTR=12e3;assert(STACK_BASE%16===0,"stack must start aligned");assert(DYNAMIC_BASE%16===0,"heap must start aligned");if(ENVIRONMENT_IS_PTHREAD){STACK_MAX=STACKTOP=STACK_MAX=2147483647;}var TOTAL_STACK=5242880;if(Module["TOTAL_STACK"])assert(TOTAL_STACK===Module["TOTAL_STACK"],"the stack size can no longer be determined at runtime");var INITIAL_INITIAL_MEMORY=Module["INITIAL_MEMORY"]||16777216;if(!Object.getOwnPropertyDescriptor(Module,"INITIAL_MEMORY"))Object.defineProperty(Module,"INITIAL_MEMORY",{configurable:true,get:function(){abort("Module.INITIAL_MEMORY has been replaced with plain INITIAL_INITIAL_MEMORY");}});assert(INITIAL_INITIAL_MEMORY>=TOTAL_STACK,"INITIAL_MEMORY should be larger than TOTAL_STACK, was "+INITIAL_INITIAL_MEMORY+"! (TOTAL_STACK="+TOTAL_STACK+")");assert(typeof Int32Array!=="undefined"&&typeof Float64Array!=="undefined"&&Int32Array.prototype.subarray!==undefined&&Int32Array.prototype.set!==undefined,"JS engine does not provide full typed array support");if(ENVIRONMENT_IS_PTHREAD){wasmMemory=Module["wasmMemory"];buffer=Module["buffer"];}else{if(Module["wasmMemory"]){wasmMemory=Module["wasmMemory"];}else{wasmMemory=new WebAssembly.Memory({"initial":INITIAL_INITIAL_MEMORY/WASM_PAGE_SIZE,"maximum":1073741824/WASM_PAGE_SIZE,"shared":true});if(!(wasmMemory.buffer instanceof SharedArrayBuffer)){err("requested a shared WebAssembly.Memory but the returned buffer is not a SharedArrayBuffer, indicating that while the browser has SharedArrayBuffer it does not have WebAssembly threads support - you may need to set a flag");if(ENVIRONMENT_IS_NODE){console.log("(on node you may need: --experimental-wasm-threads --experimental-wasm-bulk-memory and also use a recent version)");}throw Error("bad memory")}}}if(wasmMemory){buffer=wasmMemory.buffer;}INITIAL_INITIAL_MEMORY=buffer.byteLength;assert(INITIAL_INITIAL_MEMORY%WASM_PAGE_SIZE===0);assert(65536%WASM_PAGE_SIZE===0);updateGlobalBufferAndViews(buffer);if(!ENVIRONMENT_IS_PTHREAD){GROWABLE_HEAP_I32()[DYNAMICTOP_PTR>>2]=DYNAMIC_BASE;}function writeStackCookie(){assert((STACK_MAX&3)==0);GROWABLE_HEAP_U32()[(STACK_MAX>>2)+1]=34821223;GROWABLE_HEAP_U32()[(STACK_MAX>>2)+2]=2310721022;GROWABLE_HEAP_I32()[0]=1668509029;}function checkStackCookie(){var cookie1=GROWABLE_HEAP_U32()[(STACK_MAX>>2)+1];var cookie2=GROWABLE_HEAP_U32()[(STACK_MAX>>2)+2];if(cookie1!=34821223||cookie2!=2310721022){abort("Stack overflow! Stack cookie has been overwritten, expected hex dwords 0x89BACDFE and 0x2135467, but received 0x"+cookie2.toString(16)+" "+cookie1.toString(16));}if(GROWABLE_HEAP_I32()[0]!==1668509029)abort("Runtime error: The application has corrupted its heap memory area (address zero)!");}function abortStackOverflow(allocSize){abort("Stack overflow! Attempted to allocate "+allocSize+" bytes on the stack, but stack has only "+(STACK_MAX-stackSave()+allocSize)+" bytes available!");}(function(){var h16=new Int16Array(1);var h8=new Int8Array(h16.buffer);h16[0]=25459;if(h8[0]!==115||h8[1]!==99)throw "Runtime error: expected the system to be little-endian!"})();function callRuntimeCallbacks(callbacks){while(callbacks.length>0){var callback=callbacks.shift();if(typeof callback=="function"){callback(Module);continue}var func=callback.func;if(typeof func==="number"){if(callback.arg===undefined){Module["dynCall_v"](func);}else{Module["dynCall_vi"](func,callback.arg);}}else{func(callback.arg===undefined?null:callback.arg);}}}var __ATPRERUN__=[];var __ATINIT__=[];var __ATMAIN__=[];var __ATPOSTRUN__=[];var runtimeInitialized=false;var runtimeExited=false;if(ENVIRONMENT_IS_PTHREAD)runtimeInitialized=true;function preRun(){if(ENVIRONMENT_IS_PTHREAD)return;if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift());}}callRuntimeCallbacks(__ATPRERUN__);}function initRuntime(){checkStackCookie();assert(!runtimeInitialized);runtimeInitialized=true;callRuntimeCallbacks(__ATINIT__);}function preMain(){checkStackCookie();if(ENVIRONMENT_IS_PTHREAD)return;callRuntimeCallbacks(__ATMAIN__);}function postRun(){checkStackCookie();if(ENVIRONMENT_IS_PTHREAD)return;if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift());}}callRuntimeCallbacks(__ATPOSTRUN__);}function addOnPreRun(cb){__ATPRERUN__.unshift(cb);}function addOnPostRun(cb){__ATPOSTRUN__.unshift(cb);}assert(Math.imul,"This browser does not support Math.imul(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill");assert(Math.fround,"This browser does not support Math.fround(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill");assert(Math.clz32,"This browser does not support Math.clz32(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill");assert(Math.trunc,"This browser does not support Math.trunc(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill");var Math_ceil=Math.ceil;var Math_floor=Math.floor;var runDependencies=0;var runDependencyWatcher=null;var dependenciesFulfilled=null;var runDependencyTracking={};function addRunDependency(id){assert(!ENVIRONMENT_IS_PTHREAD,"addRunDependency cannot be used in a pthread worker");runDependencies++;if(Module["monitorRunDependencies"]){Module["monitorRunDependencies"](runDependencies);}if(id){assert(!runDependencyTracking[id]);runDependencyTracking[id]=1;if(runDependencyWatcher===null&&typeof setInterval!=="undefined"){runDependencyWatcher=setInterval(function(){if(ABORT){clearInterval(runDependencyWatcher);runDependencyWatcher=null;return}var shown=false;for(var dep in runDependencyTracking){if(!shown){shown=true;err("still waiting on run dependencies:");}err("dependency: "+dep);}if(shown){err("(end of list)");}},1e4);}}else{err("warning: run dependency added without ID");}}function removeRunDependency(id){runDependencies--;if(Module["monitorRunDependencies"]){Module["monitorRunDependencies"](runDependencies);}if(id){assert(runDependencyTracking[id]);delete runDependencyTracking[id];}else{err("warning: run dependency removed without ID");}if(runDependencies==0){if(runDependencyWatcher!==null){clearInterval(runDependencyWatcher);runDependencyWatcher=null;}if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback();}}}Module["preloadedImages"]={};Module["preloadedAudios"]={};function abort(what){if(Module["onAbort"]){Module["onAbort"](what);}if(ENVIRONMENT_IS_PTHREAD)console.error("Pthread aborting at "+(new Error).stack);what+="";out(what);err(what);ABORT=true;var output="abort("+what+") at "+stackTrace();what=output;throw new WebAssembly.RuntimeError(what)}var FS={error:function(){abort("Filesystem support (FS) was not included. The problem is that you are using files from JS, but files were not used from C/C++, so filesystem support was not auto-included. You can force-include filesystem support with  -s FORCE_FILESYSTEM=1");},init:function(){FS.error();},createDataFile:function(){FS.error();},createPreloadedFile:function(){FS.error();},createLazyFile:function(){FS.error();},open:function(){FS.error();},mkdev:function(){FS.error();},registerDevice:function(){FS.error();},analyzePath:function(){FS.error();},loadFilesFromDB:function(){FS.error();},ErrnoError:function ErrnoError(){FS.error();}};Module["FS_createDataFile"]=FS.createDataFile;Module["FS_createPreloadedFile"]=FS.createPreloadedFile;function hasPrefix(str,prefix){return String.prototype.startsWith?str.startsWith(prefix):str.indexOf(prefix)===0}var dataURIPrefix="data:application/octet-stream;base64,";function isDataURI(filename){return hasPrefix(filename,dataURIPrefix)}var fileURIPrefix="file://";function isFileURI(filename){return hasPrefix(filename,fileURIPrefix)}var wasmBinaryFile="tfjs-backend-wasm.wasm";if(!isDataURI(wasmBinaryFile)){wasmBinaryFile=locateFile(wasmBinaryFile);}function getBinary(){try{if(wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(wasmBinaryFile)}else{throw "both async and sync fetching of the wasm failed"}}catch(err){abort(err);}}function getBinaryPromise(){if(!wasmBinary&&(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER)&&typeof fetch==="function"&&!isFileURI(wasmBinaryFile)){return fetch(wasmBinaryFile,{credentials:"same-origin"}).then(function(response){if(!response["ok"]){throw "failed to load wasm binary file at '"+wasmBinaryFile+"'"}return response["arrayBuffer"]()}).catch(function(){return getBinary()})}return new Promise(function(resolve,reject){resolve(getBinary());})}function createWasm(){var info={"env":asmLibraryArg,"wasi_snapshot_preview1":asmLibraryArg};function receiveInstance(instance,module){var exports=instance.exports;Module["asm"]=exports;wasmModule=module;if(!ENVIRONMENT_IS_PTHREAD){var numWorkersToLoad=PThread.unusedWorkers.length;PThread.unusedWorkers.forEach(function(w){PThread.loadWasmModuleToWorker(w,function(){if(!--numWorkersToLoad)removeRunDependency("wasm-instantiate");});});}}if(!ENVIRONMENT_IS_PTHREAD){addRunDependency("wasm-instantiate");}var trueModule=Module;function receiveInstantiatedSource(output){assert(Module===trueModule,"the Module object should not be replaced during async compilation - perhaps the order of HTML elements is wrong?");trueModule=null;receiveInstance(output["instance"],output["module"]);}function instantiateArrayBuffer(receiver){return getBinaryPromise().then(function(binary){return WebAssembly.instantiate(binary,info)}).then(receiver,function(reason){err("failed to asynchronously prepare wasm: "+reason);abort(reason);})}function instantiateAsync(){if(!wasmBinary&&typeof WebAssembly.instantiateStreaming==="function"&&!isDataURI(wasmBinaryFile)&&!isFileURI(wasmBinaryFile)&&typeof fetch==="function"){fetch(wasmBinaryFile,{credentials:"same-origin"}).then(function(response){var result=WebAssembly.instantiateStreaming(response,info);return result.then(receiveInstantiatedSource,function(reason){err("wasm streaming compile failed: "+reason);err("falling back to ArrayBuffer instantiation");instantiateArrayBuffer(receiveInstantiatedSource);})});}else{return instantiateArrayBuffer(receiveInstantiatedSource)}}if(Module["instantiateWasm"]){try{var exports=Module["instantiateWasm"](info,receiveInstance);return exports}catch(e){err("Module.instantiateWasm callback failed with error: "+e);return false}}instantiateAsync();return {}}var ASM_CONSTS={};function initPthreadsJS(){PThread.initRuntime();}if(!ENVIRONMENT_IS_PTHREAD)__ATINIT__.push({func:function(){___wasm_call_ctors();}});function demangle(func){warnOnce("warning: build with  -s DEMANGLE_SUPPORT=1  to link in libcxxabi demangling");return func}function demangleAll(text){var regex=/\b_Z[\w\d_]+/g;return text.replace(regex,function(x){var y=demangle(x);return x===y?x:y+" ["+x+"]"})}var __pthread_ptr=0;var __pthread_is_main_runtime_thread=0;var __pthread_is_main_browser_thread=0;function __register_pthread_ptr(pthreadPtr,isMainBrowserThread,isMainRuntimeThread){pthreadPtr=pthreadPtr|0;isMainBrowserThread=isMainBrowserThread|0;isMainRuntimeThread=isMainRuntimeThread|0;__pthread_ptr=pthreadPtr;__pthread_is_main_browser_thread=isMainBrowserThread;__pthread_is_main_runtime_thread=isMainRuntimeThread;}Module["__register_pthread_ptr"]=__register_pthread_ptr;var ERRNO_CODES={EPERM:63,ENOENT:44,ESRCH:71,EINTR:27,EIO:29,ENXIO:60,E2BIG:1,ENOEXEC:45,EBADF:8,ECHILD:12,EAGAIN:6,EWOULDBLOCK:6,ENOMEM:48,EACCES:2,EFAULT:21,ENOTBLK:105,EBUSY:10,EEXIST:20,EXDEV:75,ENODEV:43,ENOTDIR:54,EISDIR:31,EINVAL:28,ENFILE:41,EMFILE:33,ENOTTY:59,ETXTBSY:74,EFBIG:22,ENOSPC:51,ESPIPE:70,EROFS:69,EMLINK:34,EPIPE:64,EDOM:18,ERANGE:68,ENOMSG:49,EIDRM:24,ECHRNG:106,EL2NSYNC:156,EL3HLT:107,EL3RST:108,ELNRNG:109,EUNATCH:110,ENOCSI:111,EL2HLT:112,EDEADLK:16,ENOLCK:46,EBADE:113,EBADR:114,EXFULL:115,ENOANO:104,EBADRQC:103,EBADSLT:102,EDEADLOCK:16,EBFONT:101,ENOSTR:100,ENODATA:116,ETIME:117,ENOSR:118,ENONET:119,ENOPKG:120,EREMOTE:121,ENOLINK:47,EADV:122,ESRMNT:123,ECOMM:124,EPROTO:65,EMULTIHOP:36,EDOTDOT:125,EBADMSG:9,ENOTUNIQ:126,EBADFD:127,EREMCHG:128,ELIBACC:129,ELIBBAD:130,ELIBSCN:131,ELIBMAX:132,ELIBEXEC:133,ENOSYS:52,ENOTEMPTY:55,ENAMETOOLONG:37,ELOOP:32,EOPNOTSUPP:138,EPFNOSUPPORT:139,ECONNRESET:15,ENOBUFS:42,EAFNOSUPPORT:5,EPROTOTYPE:67,ENOTSOCK:57,ENOPROTOOPT:50,ESHUTDOWN:140,ECONNREFUSED:14,EADDRINUSE:3,ECONNABORTED:13,ENETUNREACH:40,ENETDOWN:38,ETIMEDOUT:73,EHOSTDOWN:142,EHOSTUNREACH:23,EINPROGRESS:26,EALREADY:7,EDESTADDRREQ:17,EMSGSIZE:35,EPROTONOSUPPORT:66,ESOCKTNOSUPPORT:137,EADDRNOTAVAIL:4,ENETRESET:39,EISCONN:30,ENOTCONN:53,ETOOMANYREFS:141,EUSERS:136,EDQUOT:19,ESTALE:72,ENOTSUP:138,ENOMEDIUM:148,EILSEQ:25,EOVERFLOW:61,ECANCELED:11,ENOTRECOVERABLE:56,EOWNERDEAD:62,ESTRPIPE:135};var __main_thread_futex_wait_address=12912;function _emscripten_futex_wake(addr,count){if(addr<=0||addr>GROWABLE_HEAP_I8().length||addr&3!=0||count<0)return -28;if(count==0)return 0;if(count>=2147483647)count=Infinity;var mainThreadWaitAddress=Atomics.load(GROWABLE_HEAP_I32(),__main_thread_futex_wait_address>>2);var mainThreadWoken=0;if(mainThreadWaitAddress==addr){var loadedAddr=Atomics.compareExchange(GROWABLE_HEAP_I32(),__main_thread_futex_wait_address>>2,mainThreadWaitAddress,0);if(loadedAddr==mainThreadWaitAddress){--count;mainThreadWoken=1;if(count<=0)return 1}}var ret=Atomics.notify(GROWABLE_HEAP_I32(),addr>>2,count);if(ret>=0)return ret+mainThreadWoken;throw "Atomics.notify returned an unexpected value "+ret}Module["_emscripten_futex_wake"]=_emscripten_futex_wake;function __kill_thread(pthread_ptr){if(ENVIRONMENT_IS_PTHREAD)throw "Internal Error! _kill_thread() can only ever be called from main application thread!";if(!pthread_ptr)throw "Internal Error! Null pthread_ptr in _kill_thread!";GROWABLE_HEAP_I32()[pthread_ptr+12>>2]=0;var pthread=PThread.pthreads[pthread_ptr];pthread.worker.terminate();PThread.freeThreadData(pthread);PThread.runningWorkers.splice(PThread.runningWorkers.indexOf(pthread.worker),1);pthread.worker.pthread=undefined;}function __cancel_thread(pthread_ptr){if(ENVIRONMENT_IS_PTHREAD)throw "Internal Error! _cancel_thread() can only ever be called from main application thread!";if(!pthread_ptr)throw "Internal Error! Null pthread_ptr in _cancel_thread!";var pthread=PThread.pthreads[pthread_ptr];pthread.worker.postMessage({"cmd":"cancel"});}function __cleanup_thread(pthread_ptr){if(ENVIRONMENT_IS_PTHREAD)throw "Internal Error! _cleanup_thread() can only ever be called from main application thread!";if(!pthread_ptr)throw "Internal Error! Null pthread_ptr in _cleanup_thread!";GROWABLE_HEAP_I32()[pthread_ptr+12>>2]=0;var pthread=PThread.pthreads[pthread_ptr];if(pthread){var worker=pthread.worker;PThread.returnWorkerToPool(worker);}}var PThread={MAIN_THREAD_ID:1,mainThreadInfo:{schedPolicy:0,schedPrio:0},unusedWorkers:[],runningWorkers:[],initRuntime:function(){__register_pthread_ptr(PThread.mainThreadBlock,!ENVIRONMENT_IS_WORKER,1);_emscripten_register_main_browser_thread_id(PThread.mainThreadBlock);},initMainThreadBlock:function(){assert(!ENVIRONMENT_IS_PTHREAD);var pthreadPoolSize=8;for(var i=0;i<pthreadPoolSize;++i){PThread.allocateUnusedWorker();}PThread.mainThreadBlock=12160;for(var i=0;i<232/4;++i)GROWABLE_HEAP_U32()[PThread.mainThreadBlock/4+i]=0;GROWABLE_HEAP_I32()[PThread.mainThreadBlock+12>>2]=PThread.mainThreadBlock;var headPtr=PThread.mainThreadBlock+156;GROWABLE_HEAP_I32()[headPtr>>2]=headPtr;var tlsMemory=12400;for(var i=0;i<128;++i)GROWABLE_HEAP_U32()[tlsMemory/4+i]=0;Atomics.store(GROWABLE_HEAP_U32(),PThread.mainThreadBlock+104>>2,tlsMemory);Atomics.store(GROWABLE_HEAP_U32(),PThread.mainThreadBlock+40>>2,PThread.mainThreadBlock);Atomics.store(GROWABLE_HEAP_U32(),PThread.mainThreadBlock+44>>2,42);},initWorker:function(){},pthreads:{},exitHandlers:null,setThreadStatus:function(){},runExitHandlers:function(){if(PThread.exitHandlers!==null){while(PThread.exitHandlers.length>0){PThread.exitHandlers.pop()();}PThread.exitHandlers=null;}if(ENVIRONMENT_IS_PTHREAD&&threadInfoStruct)___pthread_tsd_run_dtors();},threadExit:function(exitCode){var tb=_pthread_self();if(tb){err("Pthread 0x"+tb.toString(16)+" exited.");Atomics.store(GROWABLE_HEAP_U32(),tb+4>>2,exitCode);Atomics.store(GROWABLE_HEAP_U32(),tb+0>>2,1);Atomics.store(GROWABLE_HEAP_U32(),tb+60>>2,1);Atomics.store(GROWABLE_HEAP_U32(),tb+64>>2,0);PThread.runExitHandlers();_emscripten_futex_wake(tb+0,2147483647);__register_pthread_ptr(0,0,0);threadInfoStruct=0;if(ENVIRONMENT_IS_PTHREAD){postMessage({"cmd":"exit"});}}},threadCancel:function(){PThread.runExitHandlers();Atomics.store(GROWABLE_HEAP_U32(),threadInfoStruct+4>>2,-1);Atomics.store(GROWABLE_HEAP_U32(),threadInfoStruct+0>>2,1);_emscripten_futex_wake(threadInfoStruct+0,2147483647);threadInfoStruct=selfThreadId=0;__register_pthread_ptr(0,0,0);postMessage({"cmd":"cancelDone"});},terminateAllThreads:function(){for(var t in PThread.pthreads){var pthread=PThread.pthreads[t];if(pthread&&pthread.worker){PThread.returnWorkerToPool(pthread.worker);}}PThread.pthreads={};for(var i=0;i<PThread.unusedWorkers.length;++i){var worker=PThread.unusedWorkers[i];assert(!worker.pthread);worker.terminate();}PThread.unusedWorkers=[];for(var i=0;i<PThread.runningWorkers.length;++i){var worker=PThread.runningWorkers[i];var pthread=worker.pthread;assert(pthread,"This Worker should have a pthread it is executing");PThread.freeThreadData(pthread);worker.terminate();}PThread.runningWorkers=[];},freeThreadData:function(pthread){if(!pthread)return;if(pthread.threadInfoStruct){var tlsMemory=GROWABLE_HEAP_I32()[pthread.threadInfoStruct+104>>2];GROWABLE_HEAP_I32()[pthread.threadInfoStruct+104>>2]=0;_free(tlsMemory);_free(pthread.threadInfoStruct);}pthread.threadInfoStruct=0;if(pthread.allocatedOwnStack&&pthread.stackBase)_free(pthread.stackBase);pthread.stackBase=0;if(pthread.worker)pthread.worker.pthread=null;},returnWorkerToPool:function(worker){delete PThread.pthreads[worker.pthread.thread];PThread.unusedWorkers.push(worker);PThread.runningWorkers.splice(PThread.runningWorkers.indexOf(worker),1);PThread.freeThreadData(worker.pthread);worker.pthread=undefined;},receiveObjectTransfer:function(data){},loadWasmModuleToWorker:function(worker,onFinishedLoading){worker.onmessage=function(e){var d=e["data"];var cmd=d["cmd"];if(worker.pthread)PThread.currentProxiedOperationCallerThread=worker.pthread.threadInfoStruct;if(d["targetThread"]&&d["targetThread"]!=_pthread_self()){var thread=PThread.pthreads[d.targetThread];if(thread){thread.worker.postMessage(e.data,d["transferList"]);}else{console.error('Internal error! Worker sent a message "'+cmd+'" to target pthread '+d["targetThread"]+", but that thread no longer exists!");}PThread.currentProxiedOperationCallerThread=undefined;return}if(cmd==="processQueuedMainThreadWork"){_emscripten_main_thread_process_queued_calls();}else if(cmd==="spawnThread"){__spawn_thread(e.data);}else if(cmd==="cleanupThread"){__cleanup_thread(d["thread"]);}else if(cmd==="killThread"){__kill_thread(d["thread"]);}else if(cmd==="cancelThread"){__cancel_thread(d["thread"]);}else if(cmd==="loaded"){worker.loaded=true;if(onFinishedLoading)onFinishedLoading(worker);if(worker.runPthread){worker.runPthread();delete worker.runPthread;}}else if(cmd==="print"){out("Thread "+d["threadId"]+": "+d["text"]);}else if(cmd==="printErr"){err("Thread "+d["threadId"]+": "+d["text"]);}else if(cmd==="alert"){alert("Thread "+d["threadId"]+": "+d["text"]);}else if(cmd==="exit"){var detached=worker.pthread&&Atomics.load(GROWABLE_HEAP_U32(),worker.pthread.thread+68>>2);if(detached){PThread.returnWorkerToPool(worker);}}else if(cmd==="cancelDone"){PThread.returnWorkerToPool(worker);}else if(cmd==="objectTransfer"){PThread.receiveObjectTransfer(e.data);}else if(e.data.target==="setimmediate"){worker.postMessage(e.data);}else{err("worker sent an unknown command "+cmd);}PThread.currentProxiedOperationCallerThread=undefined;};worker.onerror=function(e){err("pthread sent an error! "+e.filename+":"+e.lineno+": "+e.message);};if(ENVIRONMENT_IS_NODE){worker.on("message",function(data){worker.onmessage({data:data});});worker.on("error",function(data){worker.onerror(data);});worker.on("exit",function(data){console.log("worker exited - TODO: update the worker queue?");});}assert(wasmMemory instanceof WebAssembly.Memory,"WebAssembly memory should have been loaded by now!");assert(wasmModule instanceof WebAssembly.Module,"WebAssembly Module should have been loaded by now!");worker.postMessage({"cmd":"load","urlOrBlob":Module["mainScriptUrlOrBlob"]||_scriptDir,"wasmMemory":wasmMemory,"wasmModule":wasmModule,"DYNAMIC_BASE":DYNAMIC_BASE,"DYNAMICTOP_PTR":DYNAMICTOP_PTR});},allocateUnusedWorker:function(){var pthreadMainJs=locateFile("tfjs-backend-wasm.worker.js");PThread.unusedWorkers.push(new Worker(pthreadMainJs));},getNewWorker:function(){if(PThread.unusedWorkers.length==0){PThread.allocateUnusedWorker();PThread.loadWasmModuleToWorker(PThread.unusedWorkers[0]);}if(PThread.unusedWorkers.length>0)return PThread.unusedWorkers.pop();else return null},busySpinWait:function(msecs){var t=performance.now()+msecs;while(performance.now()<t){}}};function establishStackSpace(stackTop,stackMax){STACK_BASE=STACKTOP=stackTop;STACK_MAX=stackMax;___set_stack_limit(STACK_MAX);writeStackCookie();stackRestore(stackTop);}Module["establishStackSpace"]=establishStackSpace;function getNoExitRuntime(){return noExitRuntime}Module["getNoExitRuntime"]=getNoExitRuntime;function jsStackTrace(){var err=new Error;if(!err.stack){try{throw new Error}catch(e){err=e;}if(!err.stack){return "(no stack trace available)"}}return err.stack.toString()}function stackTrace(){var js=jsStackTrace();if(Module["extraStackTrace"])js+="\n"+Module["extraStackTrace"]();return demangleAll(js)}function ___assert_fail(condition,filename,line,func){abort("Assertion failed: "+UTF8ToString(condition)+", at: "+[filename?UTF8ToString(filename):"unknown filename",line,func?UTF8ToString(func):"unknown function"]);}var _emscripten_get_now;if(ENVIRONMENT_IS_NODE){_emscripten_get_now=function(){var t=process["hrtime"]();return t[0]*1e3+t[1]/1e6};}else if(ENVIRONMENT_IS_PTHREAD){_emscripten_get_now=function(){return performance.now()-Module["__performance_now_clock_drift"]};}else if(typeof dateNow!=="undefined"){_emscripten_get_now=dateNow;}else _emscripten_get_now=function(){return performance.now()};function _atexit(func,arg){if(ENVIRONMENT_IS_PTHREAD)return _emscripten_proxy_to_main_thread_js(1,1,func,arg);warnOnce("atexit() called, but EXIT_RUNTIME is not set, so atexits() will not be called. set EXIT_RUNTIME to 1 (see the FAQ)");}function ___handle_stack_overflow(){abort("stack overflow");}function __emscripten_notify_thread_queue(targetThreadId,mainThreadId){if(targetThreadId==mainThreadId){postMessage({"cmd":"processQueuedMainThreadWork"});}else if(ENVIRONMENT_IS_PTHREAD){postMessage({"targetThread":targetThreadId,"cmd":"processThreadQueue"});}else{var pthread=PThread.pthreads[targetThreadId];var worker=pthread&&pthread.worker;if(!worker){err("Cannot send message to thread with ID "+targetThreadId+", unknown thread ID!");return}worker.postMessage({"cmd":"processThreadQueue"});}return 1}function _abort(){abort();}function _emscripten_conditional_set_current_thread_status(expectedStatus,newStatus){}function _emscripten_futex_wait(addr,val,timeout){if(addr<=0||addr>GROWABLE_HEAP_I8().length||addr&3!=0)return -28;if(ENVIRONMENT_IS_WORKER){var ret=Atomics.wait(GROWABLE_HEAP_I32(),addr>>2,val,timeout);if(ret==="timed-out")return -73;if(ret==="not-equal")return -6;if(ret==="ok")return 0;throw "Atomics.wait returned an unexpected value "+ret}else{var loadedVal=Atomics.load(GROWABLE_HEAP_I32(),addr>>2);if(val!=loadedVal)return -6;var tNow=performance.now();var tEnd=tNow+timeout;Atomics.store(GROWABLE_HEAP_I32(),__main_thread_futex_wait_address>>2,addr);var ourWaitAddress=addr;while(addr==ourWaitAddress){tNow=performance.now();if(tNow>tEnd){return -73}_emscripten_main_thread_process_queued_calls();addr=Atomics.load(GROWABLE_HEAP_I32(),__main_thread_futex_wait_address>>2);}return 0}}function _emscripten_is_main_browser_thread(){return __pthread_is_main_browser_thread|0}function _emscripten_is_main_runtime_thread(){return __pthread_is_main_runtime_thread|0}function _emscripten_memcpy_big(dest,src,num){GROWABLE_HEAP_U8().copyWithin(dest,src,src+num);}function _emscripten_proxy_to_main_thread_js(index,sync){var numCallArgs=arguments.length-2;if(numCallArgs>20-1)throw "emscripten_proxy_to_main_thread_js: Too many arguments "+numCallArgs+" to proxied function idx="+index+", maximum supported is "+(20-1)+"!";var stack=stackSave();var args=stackAlloc(numCallArgs*8);var b=args>>3;for(var i=0;i<numCallArgs;i++){GROWABLE_HEAP_F64()[b+i]=arguments[2+i];}var ret=_emscripten_run_in_main_runtime_thread_js(index,numCallArgs,args,sync);stackRestore(stack);return ret}var _emscripten_receive_on_main_thread_js_callArgs=[];function readAsmConstArgs(sigPtr,buf){if(!readAsmConstArgs.array){readAsmConstArgs.array=[];}var args=readAsmConstArgs.array;args.length=0;var ch;while(ch=GROWABLE_HEAP_U8()[sigPtr++]){if(ch===100||ch===102){buf=buf+7&~7;args.push(GROWABLE_HEAP_F64()[buf>>3]);buf+=8;}else if(ch===105){buf=buf+3&~3;args.push(GROWABLE_HEAP_I32()[buf>>2]);buf+=4;}else abort("unexpected char in asm const signature "+ch);}return args}function _emscripten_receive_on_main_thread_js(index,numCallArgs,args){_emscripten_receive_on_main_thread_js_callArgs.length=numCallArgs;var b=args>>3;for(var i=0;i<numCallArgs;i++){_emscripten_receive_on_main_thread_js_callArgs[i]=GROWABLE_HEAP_F64()[b+i];}var isEmAsmConst=index<0;var func=!isEmAsmConst?proxiedFunctionTable[index]:ASM_CONSTS[-index-1];if(isEmAsmConst){var sigPtr=_emscripten_receive_on_main_thread_js_callArgs[1];var varargPtr=_emscripten_receive_on_main_thread_js_callArgs[2];var constArgs=readAsmConstArgs(sigPtr,varargPtr);return func.apply(null,constArgs)}assert(func.length==numCallArgs,"Call args mismatch in emscripten_receive_on_main_thread_js");return func.apply(null,_emscripten_receive_on_main_thread_js_callArgs)}function _emscripten_get_heap_size(){return GROWABLE_HEAP_U8().length}function emscripten_realloc_buffer(size){try{wasmMemory.grow(size-buffer.byteLength+65535>>>16);updateGlobalBufferAndViews(wasmMemory.buffer);return 1}catch(e){console.error("emscripten_realloc_buffer: Attempted to grow heap from "+buffer.byteLength+" bytes to "+size+" bytes, but got error: "+e);}}function _emscripten_resize_heap(requestedSize){var oldSize=_emscripten_get_heap_size();if(requestedSize<=oldSize){return false}var PAGE_MULTIPLE=65536;var maxHeapSize=1073741824;if(requestedSize>maxHeapSize){err("Cannot enlarge memory, asked to go up to "+requestedSize+" bytes, but the limit is "+maxHeapSize+" bytes!");return false}var minHeapSize=16777216;for(var cutDown=1;cutDown<=4;cutDown*=2){var overGrownHeapSize=oldSize*(1+.2/cutDown);overGrownHeapSize=Math.min(overGrownHeapSize,requestedSize+100663296);var newSize=Math.min(maxHeapSize,alignUp(Math.max(minHeapSize,requestedSize,overGrownHeapSize),PAGE_MULTIPLE));var replacement=emscripten_realloc_buffer(newSize);if(replacement){return true}}err("Failed to grow the heap from "+oldSize+" bytes to "+newSize+" bytes, not enough memory!");return false}var JSEvents={keyEvent:0,mouseEvent:0,wheelEvent:0,uiEvent:0,focusEvent:0,deviceOrientationEvent:0,deviceMotionEvent:0,fullscreenChangeEvent:0,pointerlockChangeEvent:0,visibilityChangeEvent:0,touchEvent:0,previousFullscreenElement:null,previousScreenX:null,previousScreenY:null,removeEventListenersRegistered:false,removeAllEventListeners:function(){for(var i=JSEvents.eventHandlers.length-1;i>=0;--i){JSEvents._removeHandler(i);}JSEvents.eventHandlers=[];JSEvents.deferredCalls=[];},registerRemoveEventListeners:function(){if(!JSEvents.removeEventListenersRegistered){JSEvents.removeEventListenersRegistered=true;}},deferredCalls:[],deferCall:function(targetFunction,precedence,argsList){function arraysHaveEqualContent(arrA,arrB){if(arrA.length!=arrB.length)return false;for(var i in arrA){if(arrA[i]!=arrB[i])return false}return true}for(var i in JSEvents.deferredCalls){var call=JSEvents.deferredCalls[i];if(call.targetFunction==targetFunction&&arraysHaveEqualContent(call.argsList,argsList)){return}}JSEvents.deferredCalls.push({targetFunction:targetFunction,precedence:precedence,argsList:argsList});JSEvents.deferredCalls.sort(function(x,y){return x.precedence<y.precedence});},removeDeferredCalls:function(targetFunction){for(var i=0;i<JSEvents.deferredCalls.length;++i){if(JSEvents.deferredCalls[i].targetFunction==targetFunction){JSEvents.deferredCalls.splice(i,1);--i;}}},canPerformEventHandlerRequests:function(){return JSEvents.inEventHandler&&JSEvents.currentEventHandler.allowsDeferredCalls},runDeferredCalls:function(){if(!JSEvents.canPerformEventHandlerRequests()){return}for(var i=0;i<JSEvents.deferredCalls.length;++i){var call=JSEvents.deferredCalls[i];JSEvents.deferredCalls.splice(i,1);--i;call.targetFunction.apply(null,call.argsList);}},inEventHandler:0,currentEventHandler:null,eventHandlers:[],removeAllHandlersOnTarget:function(target,eventTypeString){for(var i=0;i<JSEvents.eventHandlers.length;++i){if(JSEvents.eventHandlers[i].target==target&&(!eventTypeString||eventTypeString==JSEvents.eventHandlers[i].eventTypeString)){JSEvents._removeHandler(i--);}}},_removeHandler:function(i){var h=JSEvents.eventHandlers[i];h.target.removeEventListener(h.eventTypeString,h.eventListenerFunc,h.useCapture);JSEvents.eventHandlers.splice(i,1);},registerOrRemoveHandler:function(eventHandler){var jsEventHandler=function jsEventHandler(event){++JSEvents.inEventHandler;JSEvents.currentEventHandler=eventHandler;JSEvents.runDeferredCalls();eventHandler.handlerFunc(event);JSEvents.runDeferredCalls();--JSEvents.inEventHandler;};if(eventHandler.callbackfunc){eventHandler.eventListenerFunc=jsEventHandler;eventHandler.target.addEventListener(eventHandler.eventTypeString,jsEventHandler,eventHandler.useCapture);JSEvents.eventHandlers.push(eventHandler);JSEvents.registerRemoveEventListeners();}else{for(var i=0;i<JSEvents.eventHandlers.length;++i){if(JSEvents.eventHandlers[i].target==eventHandler.target&&JSEvents.eventHandlers[i].eventTypeString==eventHandler.eventTypeString){JSEvents._removeHandler(i--);}}}},queueEventHandlerOnThread_iiii:function(targetThread,eventHandlerFunc,eventTypeId,eventData,userData){var stackTop=stackSave();var varargs=stackAlloc(12);GROWABLE_HEAP_I32()[varargs>>2]=eventTypeId;GROWABLE_HEAP_I32()[varargs+4>>2]=eventData;GROWABLE_HEAP_I32()[varargs+8>>2]=userData;_emscripten_async_queue_on_thread_(targetThread,637534208,eventHandlerFunc,eventData,varargs);stackRestore(stackTop);},getTargetThreadForEventCallback:function(targetThread){switch(targetThread){case 1:return 0;case 2:return PThread.currentProxiedOperationCallerThread;default:return targetThread}},getNodeNameForTarget:function(target){if(!target)return "";if(target==window)return "#window";if(target==screen)return "#screen";return target&&target.nodeName?target.nodeName:""},fullscreenEnabled:function(){return document.fullscreenEnabled||document.webkitFullscreenEnabled}};function stringToNewUTF8(jsString){var length=lengthBytesUTF8(jsString)+1;var cString=_malloc(length);stringToUTF8(jsString,cString,length);return cString}function _emscripten_set_offscreencanvas_size_on_target_thread_js(targetThread,targetCanvas,width,height){var stackTop=stackSave();var varargs=stackAlloc(12);var targetCanvasPtr=0;if(targetCanvas){targetCanvasPtr=stringToNewUTF8(targetCanvas);}GROWABLE_HEAP_I32()[varargs>>2]=targetCanvasPtr;GROWABLE_HEAP_I32()[varargs+4>>2]=width;GROWABLE_HEAP_I32()[varargs+8>>2]=height;_emscripten_async_queue_on_thread_(targetThread,657457152,0,targetCanvasPtr,varargs);stackRestore(stackTop);}function _emscripten_set_offscreencanvas_size_on_target_thread(targetThread,targetCanvas,width,height){targetCanvas=targetCanvas?UTF8ToString(targetCanvas):"";_emscripten_set_offscreencanvas_size_on_target_thread_js(targetThread,targetCanvas,width,height);}function __maybeCStringToJsString(cString){return cString===cString+0?UTF8ToString(cString):cString}var __specialEventTargets=[0,typeof document!=="undefined"?document:0,typeof window!=="undefined"?window:0];function __findEventTarget(target){var domElement=__specialEventTargets[target]||(typeof document!=="undefined"?document.querySelector(__maybeCStringToJsString(target)):undefined);return domElement}function __findCanvasEventTarget(target){return __findEventTarget(target)}function _emscripten_set_canvas_element_size_calling_thread(target,width,height){var canvas=__findCanvasEventTarget(target);if(!canvas)return -4;if(canvas.canvasSharedPtr){GROWABLE_HEAP_I32()[canvas.canvasSharedPtr>>2]=width;GROWABLE_HEAP_I32()[canvas.canvasSharedPtr+4>>2]=height;}if(canvas.offscreenCanvas||!canvas.controlTransferredOffscreen){if(canvas.offscreenCanvas)canvas=canvas.offscreenCanvas;var autoResizeViewport=false;if(canvas.GLctxObject&&canvas.GLctxObject.GLctx){var prevViewport=canvas.GLctxObject.GLctx.getParameter(2978);autoResizeViewport=prevViewport[0]===0&&prevViewport[1]===0&&prevViewport[2]===canvas.width&&prevViewport[3]===canvas.height;}canvas.width=width;canvas.height=height;if(autoResizeViewport){canvas.GLctxObject.GLctx.viewport(0,0,width,height);}}else if(canvas.canvasSharedPtr){var targetThread=GROWABLE_HEAP_I32()[canvas.canvasSharedPtr+8>>2];_emscripten_set_offscreencanvas_size_on_target_thread(targetThread,target,width,height);return 1}else{return -4}return 0}function _emscripten_set_canvas_element_size_main_thread(target,width,height){if(ENVIRONMENT_IS_PTHREAD)return _emscripten_proxy_to_main_thread_js(2,1,target,width,height);return _emscripten_set_canvas_element_size_calling_thread(target,width,height)}function _emscripten_set_canvas_element_size(target,width,height){var canvas=__findCanvasEventTarget(target);if(canvas){return _emscripten_set_canvas_element_size_calling_thread(target,width,height)}else{return _emscripten_set_canvas_element_size_main_thread(target,width,height)}}function _emscripten_set_current_thread_status(newStatus){}function __webgl_acquireInstancedArraysExtension(ctx){var ext=ctx.getExtension("ANGLE_instanced_arrays");if(ext){ctx["vertexAttribDivisor"]=function(index,divisor){ext["vertexAttribDivisorANGLE"](index,divisor);};ctx["drawArraysInstanced"]=function(mode,first,count,primcount){ext["drawArraysInstancedANGLE"](mode,first,count,primcount);};ctx["drawElementsInstanced"]=function(mode,count,type,indices,primcount){ext["drawElementsInstancedANGLE"](mode,count,type,indices,primcount);};}}function __webgl_acquireVertexArrayObjectExtension(ctx){var ext=ctx.getExtension("OES_vertex_array_object");if(ext){ctx["createVertexArray"]=function(){return ext["createVertexArrayOES"]()};ctx["deleteVertexArray"]=function(vao){ext["deleteVertexArrayOES"](vao);};ctx["bindVertexArray"]=function(vao){ext["bindVertexArrayOES"](vao);};ctx["isVertexArray"]=function(vao){return ext["isVertexArrayOES"](vao)};}}function __webgl_acquireDrawBuffersExtension(ctx){var ext=ctx.getExtension("WEBGL_draw_buffers");if(ext){ctx["drawBuffers"]=function(n,bufs){ext["drawBuffersWEBGL"](n,bufs);};}}var GL={counter:1,lastError:0,buffers:[],mappedBuffers:{},programs:[],framebuffers:[],renderbuffers:[],textures:[],uniforms:[],shaders:[],vaos:[],contexts:{},currentContext:null,offscreenCanvases:{},timerQueriesEXT:[],programInfos:{},stringCache:{},unpackAlignment:4,init:function(){var miniTempFloatBuffer=new Float32Array(GL.MINI_TEMP_BUFFER_SIZE);for(var i=0;i<GL.MINI_TEMP_BUFFER_SIZE;i++){GL.miniTempBufferFloatViews[i]=miniTempFloatBuffer.subarray(0,i+1);}var miniTempIntBuffer=new Int32Array(GL.MINI_TEMP_BUFFER_SIZE);for(var i=0;i<GL.MINI_TEMP_BUFFER_SIZE;i++){GL.miniTempBufferIntViews[i]=miniTempIntBuffer.subarray(0,i+1);}},recordError:function recordError(errorCode){if(!GL.lastError){GL.lastError=errorCode;}},getNewId:function(table){var ret=GL.counter++;for(var i=table.length;i<ret;i++){table[i]=null;}return ret},MINI_TEMP_BUFFER_SIZE:256,miniTempBufferFloatViews:[0],miniTempBufferIntViews:[0],getSource:function(shader,count,string,length){var source="";for(var i=0;i<count;++i){var len=length?GROWABLE_HEAP_I32()[length+i*4>>2]:-1;source+=UTF8ToString(GROWABLE_HEAP_I32()[string+i*4>>2],len<0?undefined:len);}return source},createContext:function(canvas,webGLContextAttributes){var ctx=canvas.getContext("webgl",webGLContextAttributes);if(!ctx)return 0;var handle=GL.registerContext(ctx,webGLContextAttributes);return handle},registerContext:function(ctx,webGLContextAttributes){var handle=_malloc(8);GROWABLE_HEAP_I32()[handle+4>>2]=_pthread_self();var context={handle:handle,attributes:webGLContextAttributes,version:webGLContextAttributes.majorVersion,GLctx:ctx};if(ctx.canvas)ctx.canvas.GLctxObject=context;GL.contexts[handle]=context;if(typeof webGLContextAttributes.enableExtensionsByDefault==="undefined"||webGLContextAttributes.enableExtensionsByDefault){GL.initExtensions(context);}return handle},makeContextCurrent:function(contextHandle){GL.currentContext=GL.contexts[contextHandle];Module.ctx=GLctx=GL.currentContext&&GL.currentContext.GLctx;return !(contextHandle&&!GLctx)},getContext:function(contextHandle){return GL.contexts[contextHandle]},deleteContext:function(contextHandle){if(GL.currentContext===GL.contexts[contextHandle])GL.currentContext=null;if(typeof JSEvents==="object")JSEvents.removeAllHandlersOnTarget(GL.contexts[contextHandle].GLctx.canvas);if(GL.contexts[contextHandle]&&GL.contexts[contextHandle].GLctx.canvas)GL.contexts[contextHandle].GLctx.canvas.GLctxObject=undefined;_free(GL.contexts[contextHandle].handle);GL.contexts[contextHandle]=null;},initExtensions:function(context){if(!context)context=GL.currentContext;if(context.initExtensionsDone)return;context.initExtensionsDone=true;var GLctx=context.GLctx;if(context.version<2){__webgl_acquireInstancedArraysExtension(GLctx);__webgl_acquireVertexArrayObjectExtension(GLctx);__webgl_acquireDrawBuffersExtension(GLctx);}GLctx.disjointTimerQueryExt=GLctx.getExtension("EXT_disjoint_timer_query");var automaticallyEnabledExtensions=["OES_texture_float","OES_texture_half_float","OES_standard_derivatives","OES_vertex_array_object","WEBGL_compressed_texture_s3tc","WEBGL_depth_texture","OES_element_index_uint","EXT_texture_filter_anisotropic","EXT_frag_depth","WEBGL_draw_buffers","ANGLE_instanced_arrays","OES_texture_float_linear","OES_texture_half_float_linear","EXT_blend_minmax","EXT_shader_texture_lod","EXT_texture_norm16","WEBGL_compressed_texture_pvrtc","EXT_color_buffer_half_float","WEBGL_color_buffer_float","EXT_sRGB","WEBGL_compressed_texture_etc1","EXT_disjoint_timer_query","WEBGL_compressed_texture_etc","WEBGL_compressed_texture_astc","EXT_color_buffer_float","WEBGL_compressed_texture_s3tc_srgb","EXT_disjoint_timer_query_webgl2","WEBKIT_WEBGL_compressed_texture_pvrtc"];var exts=GLctx.getSupportedExtensions()||[];exts.forEach(function(ext){if(automaticallyEnabledExtensions.indexOf(ext)!=-1){GLctx.getExtension(ext);}});},populateUniformTable:function(program){var p=GL.programs[program];var ptable=GL.programInfos[program]={uniforms:{},maxUniformLength:0,maxAttributeLength:-1,maxUniformBlockNameLength:-1};var utable=ptable.uniforms;var numUniforms=GLctx.getProgramParameter(p,35718);for(var i=0;i<numUniforms;++i){var u=GLctx.getActiveUniform(p,i);var name=u.name;ptable.maxUniformLength=Math.max(ptable.maxUniformLength,name.length+1);if(name.slice(-1)=="]"){name=name.slice(0,name.lastIndexOf("["));}var loc=GLctx.getUniformLocation(p,name);if(loc){var id=GL.getNewId(GL.uniforms);utable[name]=[u.size,id];GL.uniforms[id]=loc;for(var j=1;j<u.size;++j){var n=name+"["+j+"]";loc=GLctx.getUniformLocation(p,n);id=GL.getNewId(GL.uniforms);GL.uniforms[id]=loc;}}}}};var __emscripten_webgl_power_preferences=["default","low-power","high-performance"];function _emscripten_webgl_do_create_context(target,attributes){assert(attributes);var contextAttributes={};var a=attributes>>2;contextAttributes["alpha"]=!!GROWABLE_HEAP_I32()[a+(0>>2)];contextAttributes["depth"]=!!GROWABLE_HEAP_I32()[a+(4>>2)];contextAttributes["stencil"]=!!GROWABLE_HEAP_I32()[a+(8>>2)];contextAttributes["antialias"]=!!GROWABLE_HEAP_I32()[a+(12>>2)];contextAttributes["premultipliedAlpha"]=!!GROWABLE_HEAP_I32()[a+(16>>2)];contextAttributes["preserveDrawingBuffer"]=!!GROWABLE_HEAP_I32()[a+(20>>2)];var powerPreference=GROWABLE_HEAP_I32()[a+(24>>2)];contextAttributes["powerPreference"]=__emscripten_webgl_power_preferences[powerPreference];contextAttributes["failIfMajorPerformanceCaveat"]=!!GROWABLE_HEAP_I32()[a+(28>>2)];contextAttributes.majorVersion=GROWABLE_HEAP_I32()[a+(32>>2)];contextAttributes.minorVersion=GROWABLE_HEAP_I32()[a+(36>>2)];contextAttributes.enableExtensionsByDefault=GROWABLE_HEAP_I32()[a+(40>>2)];contextAttributes.explicitSwapControl=GROWABLE_HEAP_I32()[a+(44>>2)];contextAttributes.proxyContextToMainThread=GROWABLE_HEAP_I32()[a+(48>>2)];contextAttributes.renderViaOffscreenBackBuffer=GROWABLE_HEAP_I32()[a+(52>>2)];var canvas=__findCanvasEventTarget(target);if(!canvas){return 0}if(contextAttributes.explicitSwapControl){return 0}var contextHandle=GL.createContext(canvas,contextAttributes);return contextHandle}function _emscripten_webgl_create_context(a0,a1){return _emscripten_webgl_do_create_context(a0,a1)}var SYSCALLS={mappings:{},buffers:[null,[],[]],printChar:function(stream,curr){var buffer=SYSCALLS.buffers[stream];assert(buffer);if(curr===0||curr===10){(stream===1?out:err)(UTF8ArrayToString(buffer,0));buffer.length=0;}else{buffer.push(curr);}},varargs:undefined,get:function(){assert(SYSCALLS.varargs!=undefined);SYSCALLS.varargs+=4;var ret=GROWABLE_HEAP_I32()[SYSCALLS.varargs-4>>2];return ret},getStr:function(ptr){var ret=UTF8ToString(ptr);return ret},get64:function(low,high){if(low>=0)assert(high===0);else assert(high===-1);return low}};function _fd_close(fd){if(ENVIRONMENT_IS_PTHREAD)return _emscripten_proxy_to_main_thread_js(3,1,fd);abort("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");return 0}function _fd_seek(fd,offset_low,offset_high,whence,newOffset){if(ENVIRONMENT_IS_PTHREAD)return _emscripten_proxy_to_main_thread_js(4,1,fd,offset_low,offset_high,whence,newOffset);abort("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");}function _fd_write(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return _emscripten_proxy_to_main_thread_js(5,1,fd,iov,iovcnt,pnum);var num=0;for(var i=0;i<iovcnt;i++){var ptr=GROWABLE_HEAP_I32()[iov+i*8>>2];var len=GROWABLE_HEAP_I32()[iov+(i*8+4)>>2];for(var j=0;j<len;j++){SYSCALLS.printChar(fd,GROWABLE_HEAP_U8()[ptr+j]);}num+=len;}GROWABLE_HEAP_I32()[pnum>>2]=num;return 0}function _pthread_cleanup_pop(execute){var routine=PThread.exitHandlers.pop();if(execute)routine();}function _pthread_cleanup_push(routine,arg){if(PThread.exitHandlers===null){PThread.exitHandlers=[];}PThread.exitHandlers.push(function(){dynCall_vi(routine,arg);});}function __spawn_thread(threadParams){if(ENVIRONMENT_IS_PTHREAD)throw "Internal Error! _spawn_thread() can only ever be called from main application thread!";var worker=PThread.getNewWorker();if(worker.pthread!==undefined)throw "Internal error!";if(!threadParams.pthread_ptr)throw "Internal error, no pthread ptr!";PThread.runningWorkers.push(worker);var tlsMemory=_malloc(128*4);for(var i=0;i<128;++i){GROWABLE_HEAP_I32()[tlsMemory+i*4>>2]=0;}var stackHigh=threadParams.stackBase+threadParams.stackSize;var pthread=PThread.pthreads[threadParams.pthread_ptr]={worker:worker,stackBase:threadParams.stackBase,stackSize:threadParams.stackSize,allocatedOwnStack:threadParams.allocatedOwnStack,thread:threadParams.pthread_ptr,threadInfoStruct:threadParams.pthread_ptr};var tis=pthread.threadInfoStruct>>2;Atomics.store(GROWABLE_HEAP_U32(),tis+(0>>2),0);Atomics.store(GROWABLE_HEAP_U32(),tis+(4>>2),0);Atomics.store(GROWABLE_HEAP_U32(),tis+(8>>2),0);Atomics.store(GROWABLE_HEAP_U32(),tis+(68>>2),threadParams.detached);Atomics.store(GROWABLE_HEAP_U32(),tis+(104>>2),tlsMemory);Atomics.store(GROWABLE_HEAP_U32(),tis+(48>>2),0);Atomics.store(GROWABLE_HEAP_U32(),tis+(40>>2),pthread.threadInfoStruct);Atomics.store(GROWABLE_HEAP_U32(),tis+(44>>2),42);Atomics.store(GROWABLE_HEAP_U32(),tis+(108>>2),threadParams.stackSize);Atomics.store(GROWABLE_HEAP_U32(),tis+(84>>2),threadParams.stackSize);Atomics.store(GROWABLE_HEAP_U32(),tis+(80>>2),stackHigh);Atomics.store(GROWABLE_HEAP_U32(),tis+(108+8>>2),stackHigh);Atomics.store(GROWABLE_HEAP_U32(),tis+(108+12>>2),threadParams.detached);Atomics.store(GROWABLE_HEAP_U32(),tis+(108+20>>2),threadParams.schedPolicy);Atomics.store(GROWABLE_HEAP_U32(),tis+(108+24>>2),threadParams.schedPrio);var global_libc=_emscripten_get_global_libc();var global_locale=global_libc+40;Atomics.store(GROWABLE_HEAP_U32(),tis+(176>>2),global_locale);worker.pthread=pthread;var msg={"cmd":"run","start_routine":threadParams.startRoutine,"arg":threadParams.arg,"threadInfoStruct":threadParams.pthread_ptr,"selfThreadId":threadParams.pthread_ptr,"parentThreadId":threadParams.parent_pthread_ptr,"stackBase":threadParams.stackBase,"stackSize":threadParams.stackSize};worker.runPthread=function(){msg.time=performance.now();worker.postMessage(msg,threadParams.transferList);};if(worker.loaded){worker.runPthread();delete worker.runPthread;}}function _pthread_getschedparam(thread,policy,schedparam){if(!policy&&!schedparam)return ERRNO_CODES.EINVAL;if(!thread){err("pthread_getschedparam called with a null thread pointer!");return ERRNO_CODES.ESRCH}var self=GROWABLE_HEAP_I32()[thread+12>>2];if(self!==thread){err("pthread_getschedparam attempted on thread "+thread+", which does not point to a valid thread, or does not exist anymore!");return ERRNO_CODES.ESRCH}var schedPolicy=Atomics.load(GROWABLE_HEAP_U32(),thread+108+20>>2);var schedPrio=Atomics.load(GROWABLE_HEAP_U32(),thread+108+24>>2);if(policy)GROWABLE_HEAP_I32()[policy>>2]=schedPolicy;if(schedparam)GROWABLE_HEAP_I32()[schedparam>>2]=schedPrio;return 0}function _pthread_self(){return __pthread_ptr|0}Module["_pthread_self"]=_pthread_self;function _pthread_create(pthread_ptr,attr,start_routine,arg){if(typeof SharedArrayBuffer==="undefined"){err("Current environment does not support SharedArrayBuffer, pthreads are not available!");return 6}if(!pthread_ptr){err("pthread_create called with a null thread pointer!");return 28}var transferList=[];var error=0;if(ENVIRONMENT_IS_PTHREAD&&(transferList.length===0||error)){return _emscripten_sync_run_in_main_thread_4(687865856,pthread_ptr,attr,start_routine,arg)}var stackSize=0;var stackBase=0;var detached=0;var schedPolicy=0;var schedPrio=0;if(attr){stackSize=GROWABLE_HEAP_I32()[attr>>2];stackSize+=81920;stackBase=GROWABLE_HEAP_I32()[attr+8>>2];detached=GROWABLE_HEAP_I32()[attr+12>>2]!==0;var inheritSched=GROWABLE_HEAP_I32()[attr+16>>2]===0;if(inheritSched){var prevSchedPolicy=GROWABLE_HEAP_I32()[attr+20>>2];var prevSchedPrio=GROWABLE_HEAP_I32()[attr+24>>2];var parentThreadPtr=PThread.currentProxiedOperationCallerThread?PThread.currentProxiedOperationCallerThread:_pthread_self();_pthread_getschedparam(parentThreadPtr,attr+20,attr+24);schedPolicy=GROWABLE_HEAP_I32()[attr+20>>2];schedPrio=GROWABLE_HEAP_I32()[attr+24>>2];GROWABLE_HEAP_I32()[attr+20>>2]=prevSchedPolicy;GROWABLE_HEAP_I32()[attr+24>>2]=prevSchedPrio;}else{schedPolicy=GROWABLE_HEAP_I32()[attr+20>>2];schedPrio=GROWABLE_HEAP_I32()[attr+24>>2];}}else{stackSize=2097152;}var allocatedOwnStack=stackBase==0;if(allocatedOwnStack){stackBase=_memalign(16,stackSize);}else{stackBase-=stackSize;assert(stackBase>0);}var threadInfoStruct=_malloc(232);for(var i=0;i<232>>2;++i)GROWABLE_HEAP_U32()[(threadInfoStruct>>2)+i]=0;GROWABLE_HEAP_I32()[pthread_ptr>>2]=threadInfoStruct;GROWABLE_HEAP_I32()[threadInfoStruct+12>>2]=threadInfoStruct;var headPtr=threadInfoStruct+156;GROWABLE_HEAP_I32()[headPtr>>2]=headPtr;var threadParams={stackBase:stackBase,stackSize:stackSize,allocatedOwnStack:allocatedOwnStack,schedPolicy:schedPolicy,schedPrio:schedPrio,detached:detached,startRoutine:start_routine,pthread_ptr:threadInfoStruct,parent_pthread_ptr:_pthread_self(),arg:arg,transferList:transferList};if(ENVIRONMENT_IS_PTHREAD){threadParams.cmd="spawnThread";postMessage(threadParams,transferList);}else{__spawn_thread(threadParams);}return 0}function _roundf(d){d=+d;return d>=+0?+Math_floor(d+ +.5):+Math_ceil(d-+.5)}if(!ENVIRONMENT_IS_PTHREAD)PThread.initMainThreadBlock();else PThread.initWorker();var GLctx;GL.init();var proxiedFunctionTable=[null,_atexit,_emscripten_set_canvas_element_size_main_thread,_fd_close,_fd_seek,_fd_write];var asmLibraryArg={"__assert_fail":___assert_fail,"__handle_stack_overflow":___handle_stack_overflow,"_emscripten_notify_thread_queue":__emscripten_notify_thread_queue,"abort":_abort,"emscripten_conditional_set_current_thread_status":_emscripten_conditional_set_current_thread_status,"emscripten_futex_wait":_emscripten_futex_wait,"emscripten_futex_wake":_emscripten_futex_wake,"emscripten_get_now":_emscripten_get_now,"emscripten_is_main_browser_thread":_emscripten_is_main_browser_thread,"emscripten_is_main_runtime_thread":_emscripten_is_main_runtime_thread,"emscripten_memcpy_big":_emscripten_memcpy_big,"emscripten_receive_on_main_thread_js":_emscripten_receive_on_main_thread_js,"emscripten_resize_heap":_emscripten_resize_heap,"emscripten_set_canvas_element_size":_emscripten_set_canvas_element_size,"emscripten_set_current_thread_status":_emscripten_set_current_thread_status,"emscripten_webgl_create_context":_emscripten_webgl_create_context,"fd_close":_fd_close,"fd_seek":_fd_seek,"fd_write":_fd_write,"initPthreadsJS":initPthreadsJS,"memory":wasmMemory||Module["wasmMemory"],"pthread_cleanup_pop":_pthread_cleanup_pop,"pthread_cleanup_push":_pthread_cleanup_push,"pthread_create":_pthread_create,"pthread_self":_pthread_self,"roundf":_roundf,"table":wasmTable};var asm=createWasm();Module["asm"]=asm;var ___wasm_call_ctors=Module["___wasm_call_ctors"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["__wasm_call_ctors"].apply(null,arguments)};var _init=Module["_init"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["init"].apply(null,arguments)};var _register_tensor=Module["_register_tensor"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["register_tensor"].apply(null,arguments)};var _dispose_data=Module["_dispose_data"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["dispose_data"].apply(null,arguments)};var _dispose=Module["_dispose"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["dispose"].apply(null,arguments)};var _Abs=Module["_Abs"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Abs"].apply(null,arguments)};var _Add=Module["_Add"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Add"].apply(null,arguments)};var _AddN=Module["_AddN"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["AddN"].apply(null,arguments)};var _ArgMax=Module["_ArgMax"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["ArgMax"].apply(null,arguments)};var _AvgPool=Module["_AvgPool"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["AvgPool"].apply(null,arguments)};var _BatchMatMul=Module["_BatchMatMul"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["BatchMatMul"].apply(null,arguments)};var _ClipByValue=Module["_ClipByValue"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["ClipByValue"].apply(null,arguments)};var _Conv2D=Module["_Conv2D"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Conv2D"].apply(null,arguments)};var _Cos=Module["_Cos"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Cos"].apply(null,arguments)};var _CropAndResize=Module["_CropAndResize"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["CropAndResize"].apply(null,arguments)};var _DepthwiseConv2dNative=Module["_DepthwiseConv2dNative"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["DepthwiseConv2dNative"].apply(null,arguments)};var _Div=Module["_Div"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Div"].apply(null,arguments)};var _Exp=Module["_Exp"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Exp"].apply(null,arguments)};var _FloorDiv=Module["_FloorDiv"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["FloorDiv"].apply(null,arguments)};var _FusedBatchNorm=Module["_FusedBatchNorm"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["FusedBatchNorm"].apply(null,arguments)};var _FusedConv2D=Module["_FusedConv2D"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["FusedConv2D"].apply(null,arguments)};var _FusedDepthwiseConv2D=Module["_FusedDepthwiseConv2D"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["FusedDepthwiseConv2D"].apply(null,arguments)};var _Gather=Module["_Gather"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Gather"].apply(null,arguments)};var _GatherNd=Module["_GatherNd"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["GatherNd"].apply(null,arguments)};var _Greater=Module["_Greater"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Greater"].apply(null,arguments)};var _GreaterEqual=Module["_GreaterEqual"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["GreaterEqual"].apply(null,arguments)};var _Less=Module["_Less"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Less"].apply(null,arguments)};var _LessEqual=Module["_LessEqual"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["LessEqual"].apply(null,arguments)};var _Log=Module["_Log"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Log"].apply(null,arguments)};var _LogicalAnd=Module["_LogicalAnd"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["LogicalAnd"].apply(null,arguments)};var _Max=Module["_Max"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Max"].apply(null,arguments)};var _MaxPool=Module["_MaxPool"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["MaxPool"].apply(null,arguments)};var _Maximum=Module["_Maximum"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Maximum"].apply(null,arguments)};var _Min=Module["_Min"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Min"].apply(null,arguments)};var _Minimum=Module["_Minimum"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Minimum"].apply(null,arguments)};var _Mul=Module["_Mul"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Mul"].apply(null,arguments)};var _Neg=Module["_Neg"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Neg"].apply(null,arguments)};var _NonMaxSuppressionV3=Module["_NonMaxSuppressionV3"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["NonMaxSuppressionV3"].apply(null,arguments)};var _NonMaxSuppressionV5=Module["_NonMaxSuppressionV5"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["NonMaxSuppressionV5"].apply(null,arguments)};var _NotEqual=Module["_NotEqual"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["NotEqual"].apply(null,arguments)};var _PadV2=Module["_PadV2"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["PadV2"].apply(null,arguments)};var _Pow=Module["_Pow"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Pow"].apply(null,arguments)};var _Prelu=Module["_Prelu"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Prelu"].apply(null,arguments)};var _Relu=Module["_Relu"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Relu"].apply(null,arguments)};var _Relu6=Module["_Relu6"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Relu6"].apply(null,arguments)};var _ResizeBilinear=Module["_ResizeBilinear"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["ResizeBilinear"].apply(null,arguments)};var _Rsqrt=Module["_Rsqrt"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Rsqrt"].apply(null,arguments)};var _ScatterNd=Module["_ScatterNd"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["ScatterNd"].apply(null,arguments)};var _Sigmoid=Module["_Sigmoid"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Sigmoid"].apply(null,arguments)};var _Sin=Module["_Sin"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Sin"].apply(null,arguments)};var _Softmax=Module["_Softmax"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Softmax"].apply(null,arguments)};var _Square=Module["_Square"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Square"].apply(null,arguments)};var _Sub=Module["_Sub"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Sub"].apply(null,arguments)};var _Sum=Module["_Sum"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Sum"].apply(null,arguments)};var _Tanh=Module["_Tanh"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Tanh"].apply(null,arguments)};var _Tile=Module["_Tile"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Tile"].apply(null,arguments)};var _Transpose=Module["_Transpose"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["Transpose"].apply(null,arguments)};var __FusedMatMul=Module["__FusedMatMul"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["_FusedMatMul"].apply(null,arguments)};var _malloc=Module["_malloc"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["malloc"].apply(null,arguments)};var _free=Module["_free"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["free"].apply(null,arguments)};var ___em_js__initPthreadsJS=Module["___em_js__initPthreadsJS"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["__em_js__initPthreadsJS"].apply(null,arguments)};var _emscripten_get_global_libc=Module["_emscripten_get_global_libc"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_get_global_libc"].apply(null,arguments)};var _memalign=Module["_memalign"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["memalign"].apply(null,arguments)};var ___pthread_tsd_run_dtors=Module["___pthread_tsd_run_dtors"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["__pthread_tsd_run_dtors"].apply(null,arguments)};var _emscripten_main_thread_process_queued_calls=Module["_emscripten_main_thread_process_queued_calls"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_main_thread_process_queued_calls"].apply(null,arguments)};var _emscripten_current_thread_process_queued_calls=Module["_emscripten_current_thread_process_queued_calls"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_current_thread_process_queued_calls"].apply(null,arguments)};var _emscripten_register_main_browser_thread_id=Module["_emscripten_register_main_browser_thread_id"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_register_main_browser_thread_id"].apply(null,arguments)};var _emscripten_main_browser_thread_id=Module["_emscripten_main_browser_thread_id"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_main_browser_thread_id"].apply(null,arguments)};var _emscripten_async_run_in_main_thread=Module["_emscripten_async_run_in_main_thread"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_async_run_in_main_thread"].apply(null,arguments)};var _emscripten_sync_run_in_main_thread=Module["_emscripten_sync_run_in_main_thread"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_sync_run_in_main_thread"].apply(null,arguments)};var _emscripten_sync_run_in_main_thread_0=Module["_emscripten_sync_run_in_main_thread_0"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_sync_run_in_main_thread_0"].apply(null,arguments)};var _emscripten_sync_run_in_main_thread_1=Module["_emscripten_sync_run_in_main_thread_1"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_sync_run_in_main_thread_1"].apply(null,arguments)};var _emscripten_sync_run_in_main_thread_2=Module["_emscripten_sync_run_in_main_thread_2"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_sync_run_in_main_thread_2"].apply(null,arguments)};var _emscripten_sync_run_in_main_thread_xprintf_varargs=Module["_emscripten_sync_run_in_main_thread_xprintf_varargs"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_sync_run_in_main_thread_xprintf_varargs"].apply(null,arguments)};var _emscripten_sync_run_in_main_thread_3=Module["_emscripten_sync_run_in_main_thread_3"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_sync_run_in_main_thread_3"].apply(null,arguments)};var _emscripten_sync_run_in_main_thread_4=Module["_emscripten_sync_run_in_main_thread_4"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_sync_run_in_main_thread_4"].apply(null,arguments)};var _emscripten_sync_run_in_main_thread_5=Module["_emscripten_sync_run_in_main_thread_5"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_sync_run_in_main_thread_5"].apply(null,arguments)};var _emscripten_sync_run_in_main_thread_6=Module["_emscripten_sync_run_in_main_thread_6"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_sync_run_in_main_thread_6"].apply(null,arguments)};var _emscripten_sync_run_in_main_thread_7=Module["_emscripten_sync_run_in_main_thread_7"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_sync_run_in_main_thread_7"].apply(null,arguments)};var _emscripten_run_in_main_runtime_thread_js=Module["_emscripten_run_in_main_runtime_thread_js"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_run_in_main_runtime_thread_js"].apply(null,arguments)};var _emscripten_async_queue_on_thread_=Module["_emscripten_async_queue_on_thread_"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_async_queue_on_thread_"].apply(null,arguments)};var _emscripten_tls_init=Module["_emscripten_tls_init"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["emscripten_tls_init"].apply(null,arguments)};var ___set_stack_limit=Module["___set_stack_limit"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["__set_stack_limit"].apply(null,arguments)};var stackSave=Module["stackSave"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["stackSave"].apply(null,arguments)};var stackAlloc=Module["stackAlloc"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["stackAlloc"].apply(null,arguments)};var stackRestore=Module["stackRestore"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["stackRestore"].apply(null,arguments)};var dynCall_vi=Module["dynCall_vi"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["dynCall_vi"].apply(null,arguments)};var dynCall_v=Module["dynCall_v"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["dynCall_v"].apply(null,arguments)};var dynCall_ii=Module["dynCall_ii"]=function(){assert(runtimeInitialized,"you need to wait for the runtime to be ready (e.g. wait for main() to be called)");assert(!runtimeExited,"the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)");return Module["asm"]["dynCall_ii"].apply(null,arguments)};Module["asm"]=asm;if(!Object.getOwnPropertyDescriptor(Module,"intArrayFromString"))Module["intArrayFromString"]=function(){abort("'intArrayFromString' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"intArrayToString"))Module["intArrayToString"]=function(){abort("'intArrayToString' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"ccall"))Module["ccall"]=function(){abort("'ccall' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};Module["cwrap"]=cwrap;if(!Object.getOwnPropertyDescriptor(Module,"setValue"))Module["setValue"]=function(){abort("'setValue' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"getValue"))Module["getValue"]=function(){abort("'getValue' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"allocate"))Module["allocate"]=function(){abort("'allocate' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"getMemory"))Module["getMemory"]=function(){abort("'getMemory' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you");};if(!Object.getOwnPropertyDescriptor(Module,"UTF8ArrayToString"))Module["UTF8ArrayToString"]=function(){abort("'UTF8ArrayToString' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"UTF8ToString"))Module["UTF8ToString"]=function(){abort("'UTF8ToString' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"stringToUTF8Array"))Module["stringToUTF8Array"]=function(){abort("'stringToUTF8Array' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"stringToUTF8"))Module["stringToUTF8"]=function(){abort("'stringToUTF8' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"lengthBytesUTF8"))Module["lengthBytesUTF8"]=function(){abort("'lengthBytesUTF8' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"stackTrace"))Module["stackTrace"]=function(){abort("'stackTrace' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"addOnPreRun"))Module["addOnPreRun"]=function(){abort("'addOnPreRun' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"addOnInit"))Module["addOnInit"]=function(){abort("'addOnInit' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"addOnPreMain"))Module["addOnPreMain"]=function(){abort("'addOnPreMain' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"addOnExit"))Module["addOnExit"]=function(){abort("'addOnExit' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"addOnPostRun"))Module["addOnPostRun"]=function(){abort("'addOnPostRun' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"writeStringToMemory"))Module["writeStringToMemory"]=function(){abort("'writeStringToMemory' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"writeArrayToMemory"))Module["writeArrayToMemory"]=function(){abort("'writeArrayToMemory' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"writeAsciiToMemory"))Module["writeAsciiToMemory"]=function(){abort("'writeAsciiToMemory' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"addRunDependency"))Module["addRunDependency"]=function(){abort("'addRunDependency' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you");};if(!Object.getOwnPropertyDescriptor(Module,"removeRunDependency"))Module["removeRunDependency"]=function(){abort("'removeRunDependency' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you");};if(!Object.getOwnPropertyDescriptor(Module,"FS_createFolder"))Module["FS_createFolder"]=function(){abort("'FS_createFolder' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you");};if(!Object.getOwnPropertyDescriptor(Module,"FS_createPath"))Module["FS_createPath"]=function(){abort("'FS_createPath' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you");};if(!Object.getOwnPropertyDescriptor(Module,"FS_createDataFile"))Module["FS_createDataFile"]=function(){abort("'FS_createDataFile' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you");};if(!Object.getOwnPropertyDescriptor(Module,"FS_createPreloadedFile"))Module["FS_createPreloadedFile"]=function(){abort("'FS_createPreloadedFile' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you");};if(!Object.getOwnPropertyDescriptor(Module,"FS_createLazyFile"))Module["FS_createLazyFile"]=function(){abort("'FS_createLazyFile' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you");};if(!Object.getOwnPropertyDescriptor(Module,"FS_createLink"))Module["FS_createLink"]=function(){abort("'FS_createLink' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you");};if(!Object.getOwnPropertyDescriptor(Module,"FS_createDevice"))Module["FS_createDevice"]=function(){abort("'FS_createDevice' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you");};if(!Object.getOwnPropertyDescriptor(Module,"FS_unlink"))Module["FS_unlink"]=function(){abort("'FS_unlink' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you");};if(!Object.getOwnPropertyDescriptor(Module,"dynamicAlloc"))Module["dynamicAlloc"]=function(){abort("'dynamicAlloc' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"loadDynamicLibrary"))Module["loadDynamicLibrary"]=function(){abort("'loadDynamicLibrary' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"loadWebAssemblyModule"))Module["loadWebAssemblyModule"]=function(){abort("'loadWebAssemblyModule' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"getLEB"))Module["getLEB"]=function(){abort("'getLEB' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"getFunctionTables"))Module["getFunctionTables"]=function(){abort("'getFunctionTables' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"alignFunctionTables"))Module["alignFunctionTables"]=function(){abort("'alignFunctionTables' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"registerFunctions"))Module["registerFunctions"]=function(){abort("'registerFunctions' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"addFunction"))Module["addFunction"]=function(){abort("'addFunction' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"removeFunction"))Module["removeFunction"]=function(){abort("'removeFunction' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"getFuncWrapper"))Module["getFuncWrapper"]=function(){abort("'getFuncWrapper' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"prettyPrint"))Module["prettyPrint"]=function(){abort("'prettyPrint' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"makeBigInt"))Module["makeBigInt"]=function(){abort("'makeBigInt' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"dynCall"))Module["dynCall"]=function(){abort("'dynCall' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"getCompilerSetting"))Module["getCompilerSetting"]=function(){abort("'getCompilerSetting' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"print"))Module["print"]=function(){abort("'print' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"printErr"))Module["printErr"]=function(){abort("'printErr' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"getTempRet0"))Module["getTempRet0"]=function(){abort("'getTempRet0' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"setTempRet0"))Module["setTempRet0"]=function(){abort("'setTempRet0' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"callMain"))Module["callMain"]=function(){abort("'callMain' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"abort"))Module["abort"]=function(){abort("'abort' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"stringToNewUTF8"))Module["stringToNewUTF8"]=function(){abort("'stringToNewUTF8' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"emscripten_realloc_buffer"))Module["emscripten_realloc_buffer"]=function(){abort("'emscripten_realloc_buffer' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"ENV"))Module["ENV"]=function(){abort("'ENV' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"setjmpId"))Module["setjmpId"]=function(){abort("'setjmpId' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"ERRNO_CODES"))Module["ERRNO_CODES"]=function(){abort("'ERRNO_CODES' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"ERRNO_MESSAGES"))Module["ERRNO_MESSAGES"]=function(){abort("'ERRNO_MESSAGES' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"setErrNo"))Module["setErrNo"]=function(){abort("'setErrNo' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"DNS"))Module["DNS"]=function(){abort("'DNS' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"GAI_ERRNO_MESSAGES"))Module["GAI_ERRNO_MESSAGES"]=function(){abort("'GAI_ERRNO_MESSAGES' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"Protocols"))Module["Protocols"]=function(){abort("'Protocols' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"Sockets"))Module["Sockets"]=function(){abort("'Sockets' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"UNWIND_CACHE"))Module["UNWIND_CACHE"]=function(){abort("'UNWIND_CACHE' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"readAsmConstArgs"))Module["readAsmConstArgs"]=function(){abort("'readAsmConstArgs' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"jstoi_q"))Module["jstoi_q"]=function(){abort("'jstoi_q' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"jstoi_s"))Module["jstoi_s"]=function(){abort("'jstoi_s' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"PATH"))Module["PATH"]=function(){abort("'PATH' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"PATH_FS"))Module["PATH_FS"]=function(){abort("'PATH_FS' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"SYSCALLS"))Module["SYSCALLS"]=function(){abort("'SYSCALLS' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"syscallMmap2"))Module["syscallMmap2"]=function(){abort("'syscallMmap2' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"syscallMunmap"))Module["syscallMunmap"]=function(){abort("'syscallMunmap' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"flush_NO_FILESYSTEM"))Module["flush_NO_FILESYSTEM"]=function(){abort("'flush_NO_FILESYSTEM' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"JSEvents"))Module["JSEvents"]=function(){abort("'JSEvents' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"demangle"))Module["demangle"]=function(){abort("'demangle' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"demangleAll"))Module["demangleAll"]=function(){abort("'demangleAll' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"jsStackTrace"))Module["jsStackTrace"]=function(){abort("'jsStackTrace' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"stackTrace"))Module["stackTrace"]=function(){abort("'stackTrace' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"getEnvStrings"))Module["getEnvStrings"]=function(){abort("'getEnvStrings' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"writeI53ToI64"))Module["writeI53ToI64"]=function(){abort("'writeI53ToI64' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"writeI53ToI64Clamped"))Module["writeI53ToI64Clamped"]=function(){abort("'writeI53ToI64Clamped' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"writeI53ToI64Signaling"))Module["writeI53ToI64Signaling"]=function(){abort("'writeI53ToI64Signaling' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"writeI53ToU64Clamped"))Module["writeI53ToU64Clamped"]=function(){abort("'writeI53ToU64Clamped' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"writeI53ToU64Signaling"))Module["writeI53ToU64Signaling"]=function(){abort("'writeI53ToU64Signaling' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"readI53FromI64"))Module["readI53FromI64"]=function(){abort("'readI53FromI64' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"readI53FromU64"))Module["readI53FromU64"]=function(){abort("'readI53FromU64' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"convertI32PairToI53"))Module["convertI32PairToI53"]=function(){abort("'convertI32PairToI53' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"convertU32PairToI53"))Module["convertU32PairToI53"]=function(){abort("'convertU32PairToI53' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"Browser"))Module["Browser"]=function(){abort("'Browser' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"GL"))Module["GL"]=function(){abort("'GL' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"emscriptenWebGLGet"))Module["emscriptenWebGLGet"]=function(){abort("'emscriptenWebGLGet' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"emscriptenWebGLGetTexPixelData"))Module["emscriptenWebGLGetTexPixelData"]=function(){abort("'emscriptenWebGLGetTexPixelData' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"emscriptenWebGLGetUniform"))Module["emscriptenWebGLGetUniform"]=function(){abort("'emscriptenWebGLGetUniform' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"emscriptenWebGLGetVertexAttrib"))Module["emscriptenWebGLGetVertexAttrib"]=function(){abort("'emscriptenWebGLGetVertexAttrib' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"AL"))Module["AL"]=function(){abort("'AL' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"SDL_unicode"))Module["SDL_unicode"]=function(){abort("'SDL_unicode' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"SDL_ttfContext"))Module["SDL_ttfContext"]=function(){abort("'SDL_ttfContext' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"SDL_audio"))Module["SDL_audio"]=function(){abort("'SDL_audio' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"SDL"))Module["SDL"]=function(){abort("'SDL' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"SDL_gfx"))Module["SDL_gfx"]=function(){abort("'SDL_gfx' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"GLUT"))Module["GLUT"]=function(){abort("'GLUT' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"EGL"))Module["EGL"]=function(){abort("'EGL' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"GLFW_Window"))Module["GLFW_Window"]=function(){abort("'GLFW_Window' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"GLFW"))Module["GLFW"]=function(){abort("'GLFW' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"GLEW"))Module["GLEW"]=function(){abort("'GLEW' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"IDBStore"))Module["IDBStore"]=function(){abort("'IDBStore' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"runAndAbortIfError"))Module["runAndAbortIfError"]=function(){abort("'runAndAbortIfError' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};Module["PThread"]=PThread;if(!Object.getOwnPropertyDescriptor(Module,"establishStackSpace"))Module["establishStackSpace"]=function(){abort("'establishStackSpace' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"getNoExitRuntime"))Module["getNoExitRuntime"]=function(){abort("'getNoExitRuntime' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"resetPrototype"))Module["resetPrototype"]=function(){abort("'resetPrototype' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"warnOnce"))Module["warnOnce"]=function(){abort("'warnOnce' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"stackSave"))Module["stackSave"]=function(){abort("'stackSave' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"stackRestore"))Module["stackRestore"]=function(){abort("'stackRestore' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"stackAlloc"))Module["stackAlloc"]=function(){abort("'stackAlloc' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"AsciiToString"))Module["AsciiToString"]=function(){abort("'AsciiToString' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"stringToAscii"))Module["stringToAscii"]=function(){abort("'stringToAscii' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"UTF16ToString"))Module["UTF16ToString"]=function(){abort("'UTF16ToString' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"stringToUTF16"))Module["stringToUTF16"]=function(){abort("'stringToUTF16' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"lengthBytesUTF16"))Module["lengthBytesUTF16"]=function(){abort("'lengthBytesUTF16' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"UTF32ToString"))Module["UTF32ToString"]=function(){abort("'UTF32ToString' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"stringToUTF32"))Module["stringToUTF32"]=function(){abort("'stringToUTF32' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"lengthBytesUTF32"))Module["lengthBytesUTF32"]=function(){abort("'lengthBytesUTF32' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"allocateUTF8"))Module["allocateUTF8"]=function(){abort("'allocateUTF8' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};if(!Object.getOwnPropertyDescriptor(Module,"allocateUTF8OnStack"))Module["allocateUTF8OnStack"]=function(){abort("'allocateUTF8OnStack' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");};Module["writeStackCookie"]=writeStackCookie;Module["checkStackCookie"]=checkStackCookie;Module["abortStackOverflow"]=abortStackOverflow;Module["PThread"]=PThread;Module["_pthread_self"]=_pthread_self;Module["wasmMemory"]=wasmMemory;Module["ExitStatus"]=ExitStatus;if(!Object.getOwnPropertyDescriptor(Module,"ALLOC_NORMAL"))Object.defineProperty(Module,"ALLOC_NORMAL",{configurable:true,get:function(){abort("'ALLOC_NORMAL' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");}});if(!Object.getOwnPropertyDescriptor(Module,"ALLOC_STACK"))Object.defineProperty(Module,"ALLOC_STACK",{configurable:true,get:function(){abort("'ALLOC_STACK' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");}});if(!Object.getOwnPropertyDescriptor(Module,"ALLOC_DYNAMIC"))Object.defineProperty(Module,"ALLOC_DYNAMIC",{configurable:true,get:function(){abort("'ALLOC_DYNAMIC' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");}});if(!Object.getOwnPropertyDescriptor(Module,"ALLOC_NONE"))Object.defineProperty(Module,"ALLOC_NONE",{configurable:true,get:function(){abort("'ALLOC_NONE' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)");}});var calledRun;Module["then"]=function(func){if(calledRun){func(Module);}else{var old=Module["onRuntimeInitialized"];Module["onRuntimeInitialized"]=function(){if(old)old();func(Module);};}return Module};function ExitStatus(status){this.name="ExitStatus";this.message="Program terminated with exit("+status+")";this.status=status;}dependenciesFulfilled=function runCaller(){if(!calledRun)run();if(!calledRun)dependenciesFulfilled=runCaller;};function run(args){if(runDependencies>0){return}writeStackCookie();preRun();if(runDependencies>0)return;function doRun(){if(calledRun)return;calledRun=true;Module["calledRun"]=true;if(ABORT)return;initRuntime();preMain();if(Module["onRuntimeInitialized"])Module["onRuntimeInitialized"]();assert(!Module["_main"],'compiled without a main, but one is present. if you added it from JS, use Module["onRuntimeInitialized"]');postRun();}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(function(){setTimeout(function(){Module["setStatus"]("");},1);doRun();},1);}else{doRun();}checkStackCookie();}Module["run"]=run;if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].pop()();}}if(!ENVIRONMENT_IS_PTHREAD)noExitRuntime=true;if(!ENVIRONMENT_IS_PTHREAD)run();


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
  var _this = undefined;
  var WASM_PRIORITY = 2;
  var BackendWasm = /** @class */ (function (_super) {
      __extends(BackendWasm, _super);
      function BackendWasm(wasm) {
          var _this = _super.call(this) || this;
          _this.wasm = wasm;
          // 0 is reserved for null data ids.
          _this.dataIdNextNumber = 1;
          _this.wasm.tfjs.init();
          _this.dataIdMap = new tfjsCore.DataStorage(_this, tfjsCore.engine());
          return _this;
      }
      BackendWasm.prototype.write = function (values, shape, dtype) {
          var dataId = {};
          this.move(dataId, values, shape, dtype);
          return dataId;
      };
      BackendWasm.prototype.numDataIds = function () {
          return this.dataIdMap.numDataIds();
      };
      BackendWasm.prototype.time = function (f) {
          return __awaiter(this, void 0, void 0, function () {
              var start, kernelMs;
              return __generator(this, function (_a) {
                  start = tfjsCore.util.now();
                  f();
                  kernelMs = tfjsCore.util.now() - start;
                  return [2 /*return*/, { kernelMs: kernelMs }];
              });
          });
      };
      BackendWasm.prototype.move = function (dataId, values, shape, dtype) {
          var id = this.dataIdNextNumber++;
          if (dtype === 'string') {
              var stringBytes = values;
              this.dataIdMap.set(dataId, { id: id, stringBytes: stringBytes, shape: shape, dtype: dtype, memoryOffset: null });
              return;
          }
          var size = tfjsCore.util.sizeFromShape(shape);
          var numBytes = size * tfjsCore.util.bytesPerElement(dtype);
          var memoryOffset = this.wasm._malloc(numBytes);
          this.dataIdMap.set(dataId, { id: id, memoryOffset: memoryOffset, shape: shape, dtype: dtype });
          this.wasm.tfjs.registerTensor(id, size, memoryOffset);
          if (values != null) {
              this.wasm.HEAPU8.set(new Uint8Array(values.buffer, 0, numBytes), memoryOffset);
          }
      };
      BackendWasm.prototype.read = function (dataId) {
          return __awaiter(this, void 0, void 0, function () {
              return __generator(this, function (_a) {
                  return [2 /*return*/, this.readSync(dataId)];
              });
          });
      };
      BackendWasm.prototype.readSync = function (dataId) {
          var _a = this.dataIdMap.get(dataId), memoryOffset = _a.memoryOffset, dtype = _a.dtype, shape = _a.shape, stringBytes = _a.stringBytes;
          if (dtype === 'string') {
              return stringBytes;
          }
          var bytes = this.wasm.HEAPU8.slice(memoryOffset, memoryOffset + tfjsCore.util.sizeFromShape(shape) * tfjsCore.util.bytesPerElement(dtype));
          return typedArrayFromBuffer(bytes.buffer, dtype);
      };
      BackendWasm.prototype.disposeData = function (dataId) {
          var data = this.dataIdMap.get(dataId);
          this.wasm._free(data.memoryOffset);
          this.wasm.tfjs.disposeData(data.id);
          this.dataIdMap.delete(dataId);
      };
      BackendWasm.prototype.floatPrecision = function () {
          return 32;
      };
      // Returns the memory offset of a tensor. Useful for debugging and unit
      // testing.
      BackendWasm.prototype.getMemoryOffset = function (dataId) {
          return this.dataIdMap.get(dataId).memoryOffset;
      };
      BackendWasm.prototype.dispose = function () {
          this.wasm.tfjs.dispose();
          this.wasm = null;
      };
      BackendWasm.prototype.memory = function () {
          return { unreliable: false };
      };
      /**
       * Make a tensor info for the output of an op. If `memoryOffset` is not
       * present, this method allocates memory on the WASM heap. If `memoryOffset`
       * is present, the memory was allocated elsewhere (in c++) and we just record
       * the pointer where that memory lives.
       */
      BackendWasm.prototype.makeOutput = function (shape, dtype, memoryOffset) {
          var dataId;
          if (memoryOffset == null) {
              dataId = this.write(null /* values */, shape, dtype);
          }
          else {
              dataId = {};
              var id = this.dataIdNextNumber++;
              this.dataIdMap.set(dataId, { id: id, memoryOffset: memoryOffset, shape: shape, dtype: dtype });
              var size = tfjsCore.util.sizeFromShape(shape);
              this.wasm.tfjs.registerTensor(id, size, memoryOffset);
          }
          return { dataId: dataId, shape: shape, dtype: dtype };
      };
      BackendWasm.prototype.typedArrayFromHeap = function (_a) {
          var shape = _a.shape, dtype = _a.dtype, dataId = _a.dataId;
          var buffer = this.wasm.HEAPU8.buffer;
          var memoryOffset = this.dataIdMap.get(dataId).memoryOffset;
          var size = tfjsCore.util.sizeFromShape(shape);
          switch (dtype) {
              case 'float32':
                  return new Float32Array(buffer, memoryOffset, size);
              case 'int32':
                  return new Int32Array(buffer, memoryOffset, size);
              case 'bool':
                  return new Uint8Array(buffer, memoryOffset, size);
              default:
                  throw new Error("Uknown dtype " + dtype);
          }
      };
      return BackendWasm;
  }(tfjsCore.KernelBackend));
  tfjsCore.registerBackend('wasm', function () { return __awaiter(_this, void 0, void 0, function () {
      var wasm;
      return __generator(this, function (_a) {
          switch (_a.label) {
              case 0: return [4 /*yield*/, init()];
              case 1:
                  wasm = (_a.sent()).wasm;
                  return [2 /*return*/, new BackendWasm(wasm)];
          }
      });
  }); }, WASM_PRIORITY);
  function createInstantiateWasmFunc(path) {
      // tslint:disable-next-line:no-any
      return function (imports, callback) {
          tfjsCore.util.fetch(path, { credentials: 'same-origin' }).then(function (response) {
              if (!response['ok']) {
                  imports.env.a("failed to load wasm binary file at '" + path + "'");
              }
              response.arrayBuffer().then(function (binary) {
                  WebAssembly.instantiate(binary, imports).then(function (output) {
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
  function init() {
      return __awaiter(this, void 0, void 0, function () {
          return __generator(this, function (_a) {
              return [2 /*return*/, new Promise(function (resolve, reject) {
                      var factoryConfig = {};
                      if (wasmPath != null) {
                          factoryConfig.locateFile = function (path, prefix) {
                              if (path.endsWith('.wasm')) {
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
                      var wasm = tfjsBackendWasm(factoryConfig);
                      var voidReturnType = null;
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
                      var initialized = false;
                      wasm.onRuntimeInitialized = function () {
                          initialized = true;
                          initAborted = false;
                          resolve({ wasm: wasm });
                      };
                      wasm.onAbort = function () {
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
                          var rejectMsg = 'Make sure the server can serve the `.wasm` file relative to the ' +
                              'bundled js file. For more details see https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md#using-bundlers';
                          reject({ message: rejectMsg });
                      };
                  })];
          });
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
              throw new Error("Unknown dtype " + dtype);
      }
  }
  var wasmPath = null;
  var initAborted = false;
  var customFetch = false;
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
  function setWasmPath(path, usePlatformFetch) {
      if (usePlatformFetch === void 0) { usePlatformFetch = false; }
      if (initAborted) {
          throw new Error('The WASM backend was already initialized. Make sure you call ' +
              '`setWasmPath()` before you call `tf.setBackend()` or `tf.ready()`');
      }
      wasmPath = path;
      customFetch = usePlatformFetch;
  }

  /** @license See the LICENSE file. */
  // This code is auto-generated, do not modify this file!
  var version = '0.0.0';

  exports.BackendWasm = BackendWasm;
  exports.setWasmPath = setWasmPath;
  exports.version_wasm = version;

  Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=tf-backend-wasm.js.map
