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
(function (global, factory) {
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('@tensorflow/tfjs-core')) :
  typeof define === 'function' && define.amd ? define(['exports', '@tensorflow/tfjs-core'], factory) :
  (global = global || self, factory(global.tf = global.tf || {}, global.tf));
}(this, (function (exports, tfjsCore) { 'use strict';

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
  // This enum must align with the enum defined in cc/backend.h.
  var CppDType;
  (function (CppDType) {
      CppDType[CppDType["float32"] = 0] = "float32";
      CppDType[CppDType["int32"] = 1] = "int32";
      CppDType[CppDType["bool"] = 2] = "bool";
      CppDType[CppDType["string"] = 3] = "string";
      CppDType[CppDType["complex64"] = 4] = "complex64";
  })(CppDType || (CppDType = {}));

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
  function registerBinaryKernel(kernelName) {
      var wasmFunc;
      function setupFunc(backend) {
          wasmFunc = backend.wasm.cwrap(kernelName, null /* void */, [
              'number',
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
          var newShape = tfjsCore.backend_util.assertAndGetBroadcastShape(a.shape, b.shape);
          var out = backend.makeOutput(newShape, a.dtype);
          // Short-circuit zero-sized tensors.
          if (tfjsCore.util.sizeFromShape(newShape) === 0) {
              return out;
          }
          var aBroadcastDims = tfjsCore.backend_util.getBroadcastDims(a.shape, newShape);
          var bBroadcastDims = tfjsCore.backend_util.getBroadcastDims(b.shape, newShape);
          var loopsOverAllOfA = aBroadcastDims.every(function (v, i) { return v === i; });
          var loopsOverAllOfB = bBroadcastDims.every(function (v, i) { return v === i; });
          var outId = backend.dataIdMap.get(out.dataId).id;
          if (loopsOverAllOfA && loopsOverAllOfB) {
              wasmFunc(aId, bId, CppDType[a.dtype], outId);
              return out;
          }
          else {
              throw new Error('Broadcasting along inner dims is not yet supported');
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
  registerBinaryKernel('Add');

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
  function setup(backend) {
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
      setupFunc: setup
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
  function setup$1(backend) {
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
      setupFunc: setup$1,
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
  function setup$2(backend) {
      wasmBatchMatMul = backend.wasm.cwrap('BatchMatMul', null /* void */, [
          'number', 'number', 'number', 'number', 'number', 'number', 'number',
          'number', 'number', 'number', 'number', 'number', 'number'
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
      var sharedDim = transposeA ? a.shape[1] : a.shape[2];
      var leftDim = transposeA ? a.shape[2] : a.shape[1];
      var rightDim = transposeB ? b.shape[1] : b.shape[2];
      var batchDim = a.shape[0];
      var aStrides = tfjsCore.util.computeStrides(a.shape);
      var bStrides = tfjsCore.util.computeStrides(b.shape);
      var _a = transposeA ?
          [aStrides[0], 1, aStrides[1]] :
          [aStrides[0], aStrides[1], 1], aBatch = _a[0], aOuterStep = _a[1], aInnerStep = _a[2];
      var _b = transposeB ?
          [1, bStrides[1], bStrides[0]] :
          [bStrides[1], 1, bStrides[0]], bInnerStep = _b[0], bOuterStep = _b[1], bBatch = _b[2];
      var out = backend.makeOutput([batchDim, leftDim, rightDim], a.dtype);
      var outId = backend.dataIdMap.get(out.dataId).id;
      wasmBatchMatMul(aId, bId, sharedDim, leftDim, rightDim, batchDim, aBatch, aOuterStep, aInnerStep, bBatch, bOuterStep, bInnerStep, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'BatchMatMul',
      backendName: 'wasm',
      setupFunc: setup$2,
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
  var wasmClip;
  function setup$3(backend) {
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
      setupFunc: setup$3,
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
  function setup$4(backend) {
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
      setupFunc: setup$4,
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
  // Must match enum in CropAndResize.cc
  var InterpolationMethod;
  (function (InterpolationMethod) {
      InterpolationMethod[InterpolationMethod["bilinear"] = 0] = "bilinear";
      InterpolationMethod[InterpolationMethod["nearest"] = 1] = "nearest";
  })(InterpolationMethod || (InterpolationMethod = {}));
  var wasmCropAndResize;
  function setup$5(backend) {
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
      var imagesId = backend.dataIdMap.get(images.dataId).id;
      var boxesId = backend.dataIdMap.get(boxes.dataId).id;
      var boxIndId = backend.dataIdMap.get(boxInd.dataId).id;
      var out = backend.makeOutput(outShape, images.dtype);
      var outId = backend.dataIdMap.get(out.dataId).id;
      var imagesShapeBytes = new Uint8Array(new Int32Array(images.shape).buffer);
      wasmCropAndResize(imagesId, boxesId, boxIndId, numBoxes, imagesShapeBytes, cropHeight, cropWidth, InterpolationMethod[method], extrapolationValue, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'CropAndResize',
      backendName: 'wasm',
      setupFunc: setup$5,
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
  function setup$6(backend) {
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
      setupFunc: setup$6,
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
  registerBinaryKernel('Div');

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
  registerBinaryKernel('FloorDiv');

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
  function setup$7(backend) {
      wasmBatchNorm = backend.wasm.cwrap('FusedBatchNorm', null /* void */, ['number', 'number', 'number', 'number', 'number', 'number', 'number']);
  }
  function fusedBatchNorm(args) {
      var backend = args.backend, inputs = args.inputs, attrs = args.attrs;
      var varianceEpsilon = attrs.varianceEpsilon;
      var x = inputs.x, mean = inputs.mean, variance = inputs.variance, offset = inputs.offset, scale = inputs.scale;
      var xId = backend.dataIdMap.get(x.dataId).id;
      var meanId = backend.dataIdMap.get(mean.dataId).id;
      var varianceId = backend.dataIdMap.get(variance.dataId).id;
      var offsetId = offset != null ? backend.dataIdMap.get(offset.dataId).id : -1;
      var scaleId = scale != null ? backend.dataIdMap.get(scale.dataId).id : -1;
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
      kernelName: 'BatchNormalization',
      backendName: 'wasm',
      setupFunc: setup$7,
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
  function setup$8(backend) {
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
  // Must match enum in conv2d_impl.h.
  var FusableActivation;
  (function (FusableActivation) {
      FusableActivation[FusableActivation["linear"] = 0] = "linear";
      FusableActivation[FusableActivation["relu"] = 1] = "relu";
      FusableActivation[FusableActivation["relu6"] = 2] = "relu6";
      FusableActivation[FusableActivation["prelu"] = 3] = "prelu";
  })(FusableActivation || (FusableActivation = {}));
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
      var biasId = -1;
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
          -1 :
          backend.dataIdMap.get(preluActivationWeights.dataId).id;
      wasmFusedConv2d(xId, batchSize, inHeight, inWidth, filterId, filterHeight, filterWidth, biasId, padTop, padRight, padBottom, padLeft, isSamePad, dilationHeight, dilationWidth, strideHeight, strideWidth, inputChannels, outputChannels, fusedActivation, preluActivationWeightsId, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'FusedConv2D',
      backendName: 'wasm',
      setupFunc: setup$8,
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
  function setup$9(backend) {
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
  // Must match enum in conv2d_impl.h.
  var FusableActivation$1;
  (function (FusableActivation) {
      FusableActivation[FusableActivation["linear"] = 0] = "linear";
      FusableActivation[FusableActivation["relu"] = 1] = "relu";
      FusableActivation[FusableActivation["relu6"] = 2] = "relu6";
      FusableActivation[FusableActivation["prelu"] = 3] = "prelu";
  })(FusableActivation$1 || (FusableActivation$1 = {}));
  function fusedDepthwiseConv2d(args) {
      var inputs = args.inputs, attrs = args.attrs, backend = args.backend;
      var convInfo = attrs.convInfo, activation = attrs.activation;
      var fusedActivation = FusableActivation$1[activation];
      if (fusedActivation == null) {
          throw new Error(activation + " activation not yet supported for FusedDepthwiseConv2D " +
              "in the wasm backend.");
      }
      var x = inputs.x, filter = inputs.filter, bias = inputs.bias, preluActivationWeights = inputs.preluActivationWeights;
      var xId = backend.dataIdMap.get(x.dataId).id;
      var filterId = backend.dataIdMap.get(filter.dataId).id;
      var outputChannels = convInfo.outChannels;
      var biasId = -1;
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
          -1 :
          backend.dataIdMap.get(preluActivationWeights.dataId).id;
      wasmFusedDepthwiseConv2d(xId, batchSize, inHeight, inWidth, filterId, filterHeight, filterWidth, biasId, padTop, padRight, padBottom, padLeft, isSamePad, dilationHeight, dilationWidth, strideHeight, strideWidth, inputChannels, outputChannels, fusedActivation, preluActivationWeightsId, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'FusedDepthwiseConv2D',
      backendName: 'wasm',
      setupFunc: setup$9,
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
  var wasmMax;
  function setup$a(backend) {
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
      setupFunc: setup$a,
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
  var wasmMaxPool;
  function setup$b(backend) {
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
      setupFunc: setup$b,
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
  function setup$c(backend) {
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
      setupFunc: setup$c,
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
  registerBinaryKernel('Mul');

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
   * Parse the result of the c++ method, which is a data structure with two ints
   * (memOffset and size).
   */
  function parseResultStruct(backend, resOffset) {
      // The result of c++ method is a data structure with two ints (memOffset, and
      // size).
      var result = new Int32Array(backend.wasm.HEAPU8.buffer, resOffset, 2);
      var memOffset = result[0];
      var size = result[1];
      // Since the result was allocated on the heap, we have to delete it.
      backend.wasm._free(resOffset);
      return { memOffset: memOffset, size: size };
  }
  var wasmFunc$2;
  function setup$d(backend) {
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
      var _a = parseResultStruct(backend, resOffset), memOffset = _a.memOffset, size = _a.size;
      var outShape = [size];
      return backend.makeOutput(outShape, 'int32', memOffset);
  }
  tfjsCore.registerKernel({
      kernelName: 'NonMaxSuppressionV3',
      backendName: 'wasm',
      setupFunc: setup$d,
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
  var wasmPadV2;
  function setup$e(backend) {
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
      setupFunc: setup$e
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
  var wasmPrelu;
  function setup$f(backend) {
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
      setupFunc: setup$f,
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
  var wasmResizeBilinear;
  function setup$g(backend) {
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
  function cropAndResize$1(args) {
      var backend = args.backend, inputs = args.inputs, attrs = args.attrs;
      var x = inputs.x;
      var alignCorners = attrs.alignCorners, newHeight = attrs.newHeight, newWidth = attrs.newWidth;
      var _a = x.shape, batch = _a[0], oldHeight = _a[1], oldWidth = _a[2], numChannels = _a[3];
      var outShape = [batch, newHeight, newWidth, numChannels];
      var xId = backend.dataIdMap.get(x.dataId).id;
      var out = backend.makeOutput(outShape, x.dtype);
      if (tfjsCore.util.sizeFromShape(x.shape) === 0) {
          return out;
      }
      var outId = backend.dataIdMap.get(out.dataId).id;
      wasmResizeBilinear(xId, batch, oldHeight, oldWidth, numChannels, newHeight, newWidth, alignCorners ? 1 : 0, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'ResizeBilinear',
      backendName: 'wasm',
      setupFunc: setup$g,
      kernelFunc: cropAndResize$1
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
  registerUnaryKernel('Sigmoid');

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
  registerBinaryKernel('Sub');

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
  function setup$h(backend) {
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
      setupFunc: setup$h,
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
    return (
  function(WasmBackendModule) {
    WasmBackendModule = WasmBackendModule || {};

  var Module=typeof WasmBackendModule!=="undefined"?WasmBackendModule:{};var moduleOverrides={};var key;for(key in Module){if(Module.hasOwnProperty(key)){moduleOverrides[key]=Module[key];}}var arguments_=[];var thisProgram="./this.program";var quit_=function(status,toThrow){throw toThrow};var ENVIRONMENT_IS_WEB=true;var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var readBinary;{if(document.currentScript){scriptDirectory=document.currentScript.src;}if(_scriptDir){scriptDirectory=_scriptDir;}if(scriptDirectory.indexOf("blob:")!==0){scriptDirectory=scriptDirectory.substr(0,scriptDirectory.lastIndexOf("/")+1);}else{scriptDirectory="";}}var out=Module["print"]||console.log.bind(console);var err=Module["printErr"]||console.warn.bind(console);for(key in moduleOverrides){if(moduleOverrides.hasOwnProperty(key)){Module[key]=moduleOverrides[key];}}moduleOverrides=null;if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"];if(Module["quit"])quit_=Module["quit"];var wasmBinary;if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];var noExitRuntime;if(Module["noExitRuntime"])noExitRuntime=Module["noExitRuntime"];if(typeof WebAssembly!=="object"){err("no native wasm support detected");}var wasmMemory;var wasmTable=new WebAssembly.Table({"initial":79,"maximum":79+0,"element":"anyfunc"});var ABORT=false;function assert(condition,text){if(!condition){abort("Assertion failed: "+text);}}function getCFunc(ident){var func=Module["_"+ident];assert(func,"Cannot call unknown function "+ident+", make sure it is exported");return func}function ccall(ident,returnType,argTypes,args,opts){var toC={"string":function(str){var ret=0;if(str!==null&&str!==undefined&&str!==0){var len=(str.length<<2)+1;ret=stackAlloc(len);stringToUTF8(str,ret,len);}return ret},"array":function(arr){var ret=stackAlloc(arr.length);writeArrayToMemory(arr,ret);return ret}};function convertReturnValue(ret){if(returnType==="string")return UTF8ToString(ret);if(returnType==="boolean")return Boolean(ret);return ret}var func=getCFunc(ident);var cArgs=[];var stack=0;if(args){for(var i=0;i<args.length;i++){var converter=toC[argTypes[i]];if(converter){if(stack===0)stack=stackSave();cArgs[i]=converter(args[i]);}else{cArgs[i]=args[i];}}}var ret=func.apply(null,cArgs);ret=convertReturnValue(ret);if(stack!==0)stackRestore(stack);return ret}function cwrap(ident,returnType,argTypes,opts){argTypes=argTypes||[];var numericArgs=argTypes.every(function(type){return type==="number"});var numericRet=returnType!=="string";if(numericRet&&numericArgs&&!opts){return getCFunc(ident)}return function(){return ccall(ident,returnType,argTypes,arguments)}}var UTF8Decoder=typeof TextDecoder!=="undefined"?new TextDecoder("utf8"):undefined;function UTF8ArrayToString(u8Array,idx,maxBytesToRead){var endIdx=idx+maxBytesToRead;var endPtr=idx;while(u8Array[endPtr]&&!(endPtr>=endIdx))++endPtr;if(endPtr-idx>16&&u8Array.subarray&&UTF8Decoder){return UTF8Decoder.decode(u8Array.subarray(idx,endPtr))}else{var str="";while(idx<endPtr){var u0=u8Array[idx++];if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=u8Array[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=u8Array[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2;}else{u0=(u0&7)<<18|u1<<12|u2<<6|u8Array[idx++]&63;}if(u0<65536){str+=String.fromCharCode(u0);}else{var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023);}}}return str}function UTF8ToString(ptr,maxBytesToRead){return ptr?UTF8ArrayToString(HEAPU8,ptr,maxBytesToRead):""}function stringToUTF8Array(str,outU8Array,outIdx,maxBytesToWrite){if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.charCodeAt(i);if(u>=55296&&u<=57343){var u1=str.charCodeAt(++i);u=65536+((u&1023)<<10)|u1&1023;}if(u<=127){if(outIdx>=endIdx)break;outU8Array[outIdx++]=u;}else if(u<=2047){if(outIdx+1>=endIdx)break;outU8Array[outIdx++]=192|u>>6;outU8Array[outIdx++]=128|u&63;}else if(u<=65535){if(outIdx+2>=endIdx)break;outU8Array[outIdx++]=224|u>>12;outU8Array[outIdx++]=128|u>>6&63;outU8Array[outIdx++]=128|u&63;}else{if(outIdx+3>=endIdx)break;outU8Array[outIdx++]=240|u>>18;outU8Array[outIdx++]=128|u>>12&63;outU8Array[outIdx++]=128|u>>6&63;outU8Array[outIdx++]=128|u&63;}}outU8Array[outIdx]=0;return outIdx-startIdx}function stringToUTF8(str,outPtr,maxBytesToWrite){return stringToUTF8Array(str,HEAPU8,outPtr,maxBytesToWrite)}var UTF16Decoder=typeof TextDecoder!=="undefined"?new TextDecoder("utf-16le"):undefined;function writeArrayToMemory(array,buffer){HEAP8.set(array,buffer);}var WASM_PAGE_SIZE=65536;function alignUp(x,multiple){if(x%multiple>0){x+=multiple-x%multiple;}return x}var buffer,HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;function updateGlobalBufferAndViews(buf){buffer=buf;Module["HEAP8"]=HEAP8=new Int8Array(buf);Module["HEAP16"]=HEAP16=new Int16Array(buf);Module["HEAP32"]=HEAP32=new Int32Array(buf);Module["HEAPU8"]=HEAPU8=new Uint8Array(buf);Module["HEAPU16"]=HEAPU16=new Uint16Array(buf);Module["HEAPU32"]=HEAPU32=new Uint32Array(buf);Module["HEAPF32"]=HEAPF32=new Float32Array(buf);Module["HEAPF64"]=HEAPF64=new Float64Array(buf);}var DYNAMIC_BASE=5247744,DYNAMICTOP_PTR=4704;var INITIAL_TOTAL_MEMORY=Module["TOTAL_MEMORY"]||16777216;if(Module["wasmMemory"]){wasmMemory=Module["wasmMemory"];}else{wasmMemory=new WebAssembly.Memory({"initial":INITIAL_TOTAL_MEMORY/WASM_PAGE_SIZE});}if(wasmMemory){buffer=wasmMemory.buffer;}INITIAL_TOTAL_MEMORY=buffer.byteLength;updateGlobalBufferAndViews(buffer);HEAP32[DYNAMICTOP_PTR>>2]=DYNAMIC_BASE;function callRuntimeCallbacks(callbacks){while(callbacks.length>0){var callback=callbacks.shift();if(typeof callback=="function"){callback();continue}var func=callback.func;if(typeof func==="number"){if(callback.arg===undefined){Module["dynCall_v"](func);}else{Module["dynCall_vi"](func,callback.arg);}}else{func(callback.arg===undefined?null:callback.arg);}}}var __ATPRERUN__=[];var __ATINIT__=[];var __ATMAIN__=[];var __ATPOSTRUN__=[];function preRun(){if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift());}}callRuntimeCallbacks(__ATPRERUN__);}function initRuntime(){callRuntimeCallbacks(__ATINIT__);}function preMain(){callRuntimeCallbacks(__ATMAIN__);}function postRun(){if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift());}}callRuntimeCallbacks(__ATPOSTRUN__);}function addOnPreRun(cb){__ATPRERUN__.unshift(cb);}function addOnPostRun(cb){__ATPOSTRUN__.unshift(cb);}var Math_ceil=Math.ceil;var Math_floor=Math.floor;var runDependencies=0;var runDependencyWatcher=null;var dependenciesFulfilled=null;function addRunDependency(id){runDependencies++;if(Module["monitorRunDependencies"]){Module["monitorRunDependencies"](runDependencies);}}function removeRunDependency(id){runDependencies--;if(Module["monitorRunDependencies"]){Module["monitorRunDependencies"](runDependencies);}if(runDependencies==0){if(runDependencyWatcher!==null){clearInterval(runDependencyWatcher);runDependencyWatcher=null;}if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback();}}}Module["preloadedImages"]={};Module["preloadedAudios"]={};function abort(what){if(Module["onAbort"]){Module["onAbort"](what);}what+="";out(what);err(what);ABORT=true;what="abort("+what+"). Build with -s ASSERTIONS=1 for more info.";throw new WebAssembly.RuntimeError(what)}var dataURIPrefix="data:application/octet-stream;base64,";function isDataURI(filename){return String.prototype.startsWith?filename.startsWith(dataURIPrefix):filename.indexOf(dataURIPrefix)===0}var wasmBinaryFile="tfjs-backend-wasm.wasm";if(!isDataURI(wasmBinaryFile)){wasmBinaryFile=locateFile(wasmBinaryFile);}function getBinary(){try{if(wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(wasmBinaryFile)}else{throw "both async and sync fetching of the wasm failed"}}catch(err){abort(err);}}function getBinaryPromise(){if(!wasmBinary&&(ENVIRONMENT_IS_WEB)&&typeof fetch==="function"){return fetch(wasmBinaryFile,{credentials:"same-origin"}).then(function(response){if(!response["ok"]){throw "failed to load wasm binary file at '"+wasmBinaryFile+"'"}return response["arrayBuffer"]()}).catch(function(){return getBinary()})}return new Promise(function(resolve,reject){resolve(getBinary());})}function createWasm(){var info={"env":asmLibraryArg,"wasi_unstable":asmLibraryArg};function receiveInstance(instance,module){var exports=instance.exports;Module["asm"]=exports;removeRunDependency();}addRunDependency();function receiveInstantiatedSource(output){receiveInstance(output["instance"]);}function instantiateArrayBuffer(receiver){return getBinaryPromise().then(function(binary){return WebAssembly.instantiate(binary,info)}).then(receiver,function(reason){err("failed to asynchronously prepare wasm: "+reason);abort(reason);})}function instantiateAsync(){if(!wasmBinary&&typeof WebAssembly.instantiateStreaming==="function"&&!isDataURI(wasmBinaryFile)&&typeof fetch==="function"){fetch(wasmBinaryFile,{credentials:"same-origin"}).then(function(response){var result=WebAssembly.instantiateStreaming(response,info);return result.then(receiveInstantiatedSource,function(reason){err("wasm streaming compile failed: "+reason);err("falling back to ArrayBuffer instantiation");instantiateArrayBuffer(receiveInstantiatedSource);})});}else{return instantiateArrayBuffer(receiveInstantiatedSource)}}if(Module["instantiateWasm"]){try{var exports=Module["instantiateWasm"](info,receiveInstance);return exports}catch(e){err("Module.instantiateWasm callback failed with error: "+e);return false}}instantiateAsync();return {}}__ATINIT__.push({func:function(){___wasm_call_ctors();}});function _abort(){abort();}function _emscripten_memcpy_big(dest,src,num){HEAPU8.set(HEAPU8.subarray(src,src+num),dest);}function _emscripten_get_heap_size(){return HEAP8.length}function emscripten_realloc_buffer(size){try{wasmMemory.grow(size-buffer.byteLength+65535>>16);updateGlobalBufferAndViews(wasmMemory.buffer);return 1}catch(e){}}function _emscripten_resize_heap(requestedSize){var oldSize=_emscripten_get_heap_size();var PAGE_MULTIPLE=65536;var LIMIT=2147483648-PAGE_MULTIPLE;if(requestedSize>LIMIT){return false}var MIN_TOTAL_MEMORY=16777216;var newSize=Math.max(oldSize,MIN_TOTAL_MEMORY);while(newSize<requestedSize){if(newSize<=536870912){newSize=alignUp(2*newSize,PAGE_MULTIPLE);}else{newSize=Math.min(alignUp((3*newSize+2147483648)/4,PAGE_MULTIPLE),LIMIT);}}var replacement=emscripten_realloc_buffer(newSize);if(!replacement){return false}return true}var SYSCALLS={buffers:[null,[],[]],printChar:function(stream,curr){var buffer=SYSCALLS.buffers[stream];if(curr===0||curr===10){(stream===1?out:err)(UTF8ArrayToString(buffer,0));buffer.length=0;}else{buffer.push(curr);}},varargs:0,get:function(varargs){SYSCALLS.varargs+=4;var ret=HEAP32[SYSCALLS.varargs-4>>2];return ret},getStr:function(){var ret=UTF8ToString(SYSCALLS.get());return ret},get64:function(){var low=SYSCALLS.get(),high=SYSCALLS.get();return low},getZero:function(){SYSCALLS.get();}};function _fd_close(fd){try{return 0}catch(e){if(typeof FS==="undefined"||!(e instanceof FS.ErrnoError))abort(e);return e.errno}}function _fd_seek(fd,offset_low,offset_high,whence,newOffset){try{return 0}catch(e){if(typeof FS==="undefined"||!(e instanceof FS.ErrnoError))abort(e);return e.errno}}function _fd_write(fd,iov,iovcnt,pnum){try{var num=0;for(var i=0;i<iovcnt;i++){var ptr=HEAP32[iov+i*8>>2];var len=HEAP32[iov+(i*8+4)>>2];for(var j=0;j<len;j++){SYSCALLS.printChar(fd,HEAPU8[ptr+j]);}num+=len;}HEAP32[pnum>>2]=num;return 0}catch(e){if(typeof FS==="undefined"||!(e instanceof FS.ErrnoError))abort(e);return e.errno}}function _roundf(d){d=+d;return d>=+0?+Math_floor(d+ +.5):+Math_ceil(d-+.5)}var asmLibraryArg={"a":_abort,"d":_emscripten_memcpy_big,"e":_emscripten_resize_heap,"f":_fd_close,"c":_fd_seek,"g":_fd_write,"memory":wasmMemory,"b":_roundf,"table":wasmTable};var asm=createWasm();Module["asm"]=asm;var ___wasm_call_ctors=Module["___wasm_call_ctors"]=function(){return Module["asm"]["h"].apply(null,arguments)};var _init=Module["_init"]=function(){return Module["asm"]["i"].apply(null,arguments)};var _register_tensor=Module["_register_tensor"]=function(){return Module["asm"]["j"].apply(null,arguments)};var _dispose_data=Module["_dispose_data"]=function(){return Module["asm"]["k"].apply(null,arguments)};var _dispose=Module["_dispose"]=function(){return Module["asm"]["l"].apply(null,arguments)};var _Abs=Module["_Abs"]=function(){return Module["asm"]["m"].apply(null,arguments)};var _Add=Module["_Add"]=function(){return Module["asm"]["n"].apply(null,arguments)};var _AddN=Module["_AddN"]=function(){return Module["asm"]["o"].apply(null,arguments)};var _ArgMax=Module["_ArgMax"]=function(){return Module["asm"]["p"].apply(null,arguments)};var _AvgPool=Module["_AvgPool"]=function(){return Module["asm"]["q"].apply(null,arguments)};var _BatchMatMul=Module["_BatchMatMul"]=function(){return Module["asm"]["r"].apply(null,arguments)};var _ClipByValue=Module["_ClipByValue"]=function(){return Module["asm"]["s"].apply(null,arguments)};var _Conv2D=Module["_Conv2D"]=function(){return Module["asm"]["t"].apply(null,arguments)};var _CropAndResize=Module["_CropAndResize"]=function(){return Module["asm"]["u"].apply(null,arguments)};var _DepthwiseConv2dNative=Module["_DepthwiseConv2dNative"]=function(){return Module["asm"]["v"].apply(null,arguments)};var _Div=Module["_Div"]=function(){return Module["asm"]["w"].apply(null,arguments)};var _FloorDiv=Module["_FloorDiv"]=function(){return Module["asm"]["x"].apply(null,arguments)};var _FusedBatchNorm=Module["_FusedBatchNorm"]=function(){return Module["asm"]["y"].apply(null,arguments)};var _FusedConv2D=Module["_FusedConv2D"]=function(){return Module["asm"]["z"].apply(null,arguments)};var _FusedDepthwiseConv2D=Module["_FusedDepthwiseConv2D"]=function(){return Module["asm"]["A"].apply(null,arguments)};var _Max=Module["_Max"]=function(){return Module["asm"]["B"].apply(null,arguments)};var _MaxPool=Module["_MaxPool"]=function(){return Module["asm"]["C"].apply(null,arguments)};var _Min=Module["_Min"]=function(){return Module["asm"]["D"].apply(null,arguments)};var _Mul=Module["_Mul"]=function(){return Module["asm"]["E"].apply(null,arguments)};var _NonMaxSuppressionV3=Module["_NonMaxSuppressionV3"]=function(){return Module["asm"]["F"].apply(null,arguments)};var _malloc=Module["_malloc"]=function(){return Module["asm"]["G"].apply(null,arguments)};var _PadV2=Module["_PadV2"]=function(){return Module["asm"]["H"].apply(null,arguments)};var _Prelu=Module["_Prelu"]=function(){return Module["asm"]["I"].apply(null,arguments)};var _Relu=Module["_Relu"]=function(){return Module["asm"]["J"].apply(null,arguments)};var _Relu6=Module["_Relu6"]=function(){return Module["asm"]["K"].apply(null,arguments)};var _ResizeBilinear=Module["_ResizeBilinear"]=function(){return Module["asm"]["L"].apply(null,arguments)};var _Sigmoid=Module["_Sigmoid"]=function(){return Module["asm"]["M"].apply(null,arguments)};var _Square=Module["_Square"]=function(){return Module["asm"]["N"].apply(null,arguments)};var _Sub=Module["_Sub"]=function(){return Module["asm"]["O"].apply(null,arguments)};var _Transpose=Module["_Transpose"]=function(){return Module["asm"]["P"].apply(null,arguments)};var _free=Module["_free"]=function(){return Module["asm"]["Q"].apply(null,arguments)};var stackSave=Module["stackSave"]=function(){return Module["asm"]["R"].apply(null,arguments)};var stackAlloc=Module["stackAlloc"]=function(){return Module["asm"]["S"].apply(null,arguments)};var stackRestore=Module["stackRestore"]=function(){return Module["asm"]["T"].apply(null,arguments)};var dynCall_vi=Module["dynCall_vi"]=function(){return Module["asm"]["U"].apply(null,arguments)};var dynCall_v=Module["dynCall_v"]=function(){return Module["asm"]["V"].apply(null,arguments)};Module["asm"]=asm;Module["cwrap"]=cwrap;var calledRun;Module["then"]=function(func){if(calledRun){func(Module);}else{var old=Module["onRuntimeInitialized"];Module["onRuntimeInitialized"]=function(){if(old)old();func(Module);};}return Module};dependenciesFulfilled=function runCaller(){if(!calledRun)run();if(!calledRun)dependenciesFulfilled=runCaller;};function run(args){if(runDependencies>0){return}preRun();if(runDependencies>0)return;function doRun(){if(calledRun)return;calledRun=true;if(ABORT)return;initRuntime();preMain();if(Module["onRuntimeInitialized"])Module["onRuntimeInitialized"]();postRun();}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(function(){setTimeout(function(){Module["setStatus"]("");},1);doRun();},1);}else{doRun();}}Module["run"]=run;if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].pop()();}}noExitRuntime=true;run();


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
          _this.dataIdNextNumber = 0;
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
              this.wasm.HEAPU8.set(new Uint8Array(values.buffer), memoryOffset);
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
  /**
   * Initializes the wasm module and creates the js <--> wasm bridge.
   *
   * NOTE: We wrap the wasm module in a object with property 'wasm' instead of
   * returning Promise<BackendWasmModule> to avoid freezing Chrome (last tested in
   * Chrome 76).
   */
  function init() {
      return __awaiter(this, void 0, void 0, function () {
          return __generator(this, function (_a) {
              return [2 /*return*/, new Promise(function (resolve) {
                      var wasm = tfjsBackendWasm();
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
                      wasm.onRuntimeInitialized = function () { return resolve({ wasm: wasm }); };
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
              throw new Error("Uknown dtype " + dtype);
      }
  }

  exports.BackendWasm = BackendWasm;

  Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=tf-backend-wasm.js.map
