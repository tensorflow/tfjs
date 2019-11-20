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
  var wasmAvgPool;
  function setup(backend) {
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
      setupFunc: setup,
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
  var wasmBatchMatMul;
  function setup$1(backend) {
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
      setupFunc: setup$1,
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
  function setup$2(backend) {
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
      setupFunc: setup$2,
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
  function setup$3(backend) {
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
      setupFunc: setup$3,
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
  function setup$4(backend) {
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
      setupFunc: setup$4,
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
  function setup$5(backend) {
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
      setupFunc: setup$5,
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
  var wasmBatchNorm;
  function setup$6(backend) {
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
      setupFunc: setup$6,
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
  function setup$7(backend) {
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
      ]);
  }
  function fusedConv2d(args) {
      var inputs = args.inputs, attrs = args.attrs, backend = args.backend;
      var convInfo = attrs.convInfo, activation = attrs.activation;
      if (activation !== 'linear') {
          throw new Error(activation + " activation not yet supported for FusedConv2D " +
              "in the wasm backend.");
      }
      var x = inputs.x, filter = inputs.filter, bias = inputs.bias;
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
      wasmFusedConv2d(xId, batchSize, inHeight, inWidth, filterId, filterHeight, filterWidth, biasId, padTop, padRight, padBottom, padLeft, isSamePad, dilationHeight, dilationWidth, strideHeight, strideWidth, inputChannels, outputChannels, outId);
      return out;
  }
  tfjsCore.registerKernel({
      kernelName: 'FusedConv2D',
      backendName: 'wasm',
      setupFunc: setup$7,
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
  var wasmMax;
  function setup$8(backend) {
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
      setupFunc: setup$8,
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
  function setup$9(backend) {
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
      setupFunc: setup$9,
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
  function setup$a(backend) {
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
      setupFunc: setup$a,
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
  var wasmPadV2;
  function setup$b(backend) {
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
      setupFunc: setup$b
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
  function setup$c(backend) {
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
      setupFunc: setup$c,
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
  function setup$d(backend) {
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
      setupFunc: setup$d,
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

  // Copyright 2010 The Emscripten Authors.  All rights reserved.
  // Emscripten is available under two separate licenses, the MIT license and the
  // University of Illinois/NCSA Open Source License.  Both these licenses can be
  // found in the LICENSE file.

  // The Module object: Our interface to the outside world. We import
  // and export values on it. There are various ways Module can be used:
  // 1. Not defined. We create it here
  // 2. A function parameter, function(Module) { ..generated code.. }
  // 3. pre-run appended it, var Module = {}; ..generated code..
  // 4. External script tag defines var Module.
  // We need to check if Module already exists (e.g. case 3 above).
  // Substitution will be replaced with actual code on later stage of the build,
  // this way Closure Compiler will not mangle it (e.g. case 4. above).
  // Note that if you want to run closure, and also to use Module
  // after the generated code, you will need to define   var Module = {};
  // before the code. Then that object will be used in the code, and you
  // can continue to use Module afterwards as well.
  var Module = typeof WasmBackendModule !== 'undefined' ? WasmBackendModule : {};

  // --pre-jses are emitted after the Module integration code, so that they can
  // refer to Module (if they choose; they can also define Module)
  // {{PRE_JSES}}

  // Sometimes an existing Module object exists with properties
  // meant to overwrite the default module functionality. Here
  // we collect those properties and reapply _after_ we configure
  // the current environment's defaults to avoid having to be so
  // defensive during initialization.
  var moduleOverrides = {};
  var key;
  for (key in Module) {
    if (Module.hasOwnProperty(key)) {
      moduleOverrides[key] = Module[key];
    }
  }

  var arguments_ = [];
  var thisProgram = './this.program';
  var quit_ = function(status, toThrow) {
    throw toThrow;
  };

  // Determine the runtime environment we are in. You can customize this by
  // setting the ENVIRONMENT setting at compile time (see settings.js).

  var ENVIRONMENT_IS_WEB = true;

  if (Module['ENVIRONMENT']) {
    throw new Error('Module.ENVIRONMENT has been deprecated. To force the environment, use the ENVIRONMENT compile-time option (for example, -s ENVIRONMENT=web or -s ENVIRONMENT=node)');
  }



  // `/` should be present at the end if `scriptDirectory` is not empty
  var scriptDirectory = '';
  function locateFile(path) {
    if (Module['locateFile']) {
      return Module['locateFile'](path, scriptDirectory);
    }
    return scriptDirectory + path;
  }

  // Hooks that are implemented differently in different runtime environments.
  var readBinary;


  // Note that this includes Node.js workers when relevant (pthreads is enabled).
  // Node.js workers are detected as a combination of ENVIRONMENT_IS_WORKER and
  // ENVIRONMENT_HAS_NODE.
  {
    if (document.currentScript) { // web
      scriptDirectory = document.currentScript.src;
    }
    // When MODULARIZE (and not _INSTANCE), this JS may be executed later, after document.currentScript
    // is gone, so we saved it, and we use it here instead of any other info.
    if (_scriptDir) {
      scriptDirectory = _scriptDir;
    }
    // blob urls look like blob:http://site.com/etc/etc and we cannot infer anything from them.
    // otherwise, slice off the final part of the url to find the script directory.
    // if scriptDirectory does not contain a slash, lastIndexOf will return -1,
    // and scriptDirectory will correctly be replaced with an empty string.
    if (scriptDirectory.indexOf('blob:') !== 0) {
      scriptDirectory = scriptDirectory.substr(0, scriptDirectory.lastIndexOf('/')+1);
    } else {
      scriptDirectory = '';
    }

    if (!(typeof window === 'object' || typeof importScripts === 'function')) throw new Error('not compiled for this environment (did you build to HTML and try to run it not on the web, or set ENVIRONMENT to something - like node - and run it someplace else - like on the web?)');
  }


  // Set up the out() and err() hooks, which are how we can print to stdout or
  // stderr, respectively.
  var out = Module['print'] || console.log.bind(console);
  var err = Module['printErr'] || console.warn.bind(console);

  // Merge back in the overrides
  for (key in moduleOverrides) {
    if (moduleOverrides.hasOwnProperty(key)) {
      Module[key] = moduleOverrides[key];
    }
  }
  // Free the object hierarchy contained in the overrides, this lets the GC
  // reclaim data used e.g. in memoryInitializerRequest, which is a large typed array.
  moduleOverrides = null;

  // Emit code to handle expected values on the Module object. This applies Module.x
  // to the proper local x. This has two benefits: first, we only emit it if it is
  // expected to arrive, and second, by using a local everywhere else that can be
  // minified.
  if (Module['arguments']) arguments_ = Module['arguments'];if (!Object.getOwnPropertyDescriptor(Module, 'arguments')) Object.defineProperty(Module, 'arguments', { configurable: true, get: function() { abort('Module.arguments has been replaced with plain arguments_'); } });
  if (Module['thisProgram']) thisProgram = Module['thisProgram'];if (!Object.getOwnPropertyDescriptor(Module, 'thisProgram')) Object.defineProperty(Module, 'thisProgram', { configurable: true, get: function() { abort('Module.thisProgram has been replaced with plain thisProgram'); } });
  if (Module['quit']) quit_ = Module['quit'];if (!Object.getOwnPropertyDescriptor(Module, 'quit')) Object.defineProperty(Module, 'quit', { configurable: true, get: function() { abort('Module.quit has been replaced with plain quit_'); } });

  // perform assertions in shell.js after we set up out() and err(), as otherwise if an assertion fails it cannot print the message
  // Assertions on removed incoming Module JS APIs.
  assert(typeof Module['memoryInitializerPrefixURL'] === 'undefined', 'Module.memoryInitializerPrefixURL option was removed, use Module.locateFile instead');
  assert(typeof Module['pthreadMainPrefixURL'] === 'undefined', 'Module.pthreadMainPrefixURL option was removed, use Module.locateFile instead');
  assert(typeof Module['cdInitializerPrefixURL'] === 'undefined', 'Module.cdInitializerPrefixURL option was removed, use Module.locateFile instead');
  assert(typeof Module['filePackagePrefixURL'] === 'undefined', 'Module.filePackagePrefixURL option was removed, use Module.locateFile instead');
  assert(typeof Module['read'] === 'undefined', 'Module.read option was removed (modify read_ in JS)');
  assert(typeof Module['readAsync'] === 'undefined', 'Module.readAsync option was removed (modify readAsync in JS)');
  assert(typeof Module['readBinary'] === 'undefined', 'Module.readBinary option was removed (modify readBinary in JS)');
  assert(typeof Module['setWindowTitle'] === 'undefined', 'Module.setWindowTitle option was removed (modify setWindowTitle in JS)');
  if (!Object.getOwnPropertyDescriptor(Module, 'read')) Object.defineProperty(Module, 'read', { configurable: true, get: function() { abort('Module.read has been replaced with plain read_'); } });
  if (!Object.getOwnPropertyDescriptor(Module, 'readAsync')) Object.defineProperty(Module, 'readAsync', { configurable: true, get: function() { abort('Module.readAsync has been replaced with plain readAsync'); } });
  if (!Object.getOwnPropertyDescriptor(Module, 'readBinary')) Object.defineProperty(Module, 'readBinary', { configurable: true, get: function() { abort('Module.readBinary has been replaced with plain readBinary'); } });

  // stack management, and other functionality that is provided by the compiled code,
  // should not be used before it is ready
  stackSave = stackRestore = stackAlloc = function() {
    abort('cannot use the stack before compiled code is ready to run, and has provided stack access');
  };

  function warnOnce(text) {
    if (!warnOnce.shown) warnOnce.shown = {};
    if (!warnOnce.shown[text]) {
      warnOnce.shown[text] = 1;
      err(text);
    }
  }

  var tempRet0 = 0;

  var setTempRet0 = function(value) {
    tempRet0 = value;
  };

  var getTempRet0 = function() {
    return tempRet0;
  };




  // === Preamble library stuff ===

  // Documentation for the public APIs defined in this file must be updated in:
  //    site/source/docs/api_reference/preamble.js.rst
  // A prebuilt local version of the documentation is available at:
  //    site/build/text/docs/api_reference/preamble.js.txt
  // You can also build docs locally as HTML or other formats in site/
  // An online HTML version (which may be of a different version of Emscripten)
  //    is up at http://kripken.github.io/emscripten-site/docs/api_reference/preamble.js.html


  var wasmBinary;if (Module['wasmBinary']) wasmBinary = Module['wasmBinary'];if (!Object.getOwnPropertyDescriptor(Module, 'wasmBinary')) Object.defineProperty(Module, 'wasmBinary', { configurable: true, get: function() { abort('Module.wasmBinary has been replaced with plain wasmBinary'); } });
  var noExitRuntime;if (Module['noExitRuntime']) noExitRuntime = Module['noExitRuntime'];if (!Object.getOwnPropertyDescriptor(Module, 'noExitRuntime')) Object.defineProperty(Module, 'noExitRuntime', { configurable: true, get: function() { abort('Module.noExitRuntime has been replaced with plain noExitRuntime'); } });


  if (typeof WebAssembly !== 'object') {
    abort('No WebAssembly support found. Build with -s WASM=0 to target JavaScript instead.');
  }


  // In MINIMAL_RUNTIME, setValue() and getValue() are only available when building with safe heap enabled, for heap safety checking.
  // In traditional runtime, setValue() and getValue() are always available (although their use is highly discouraged due to perf penalties)

  /** @type {function(number, number, string, boolean=)} */
  function setValue(ptr, value, type, noSafe) {
    type = type || 'i8';
    if (type.charAt(type.length-1) === '*') type = 'i32'; // pointers are 32-bit
    if (noSafe) {
      switch(type) {
        case 'i1': HEAP8[((ptr)>>0)]=value; break;
        case 'i8': HEAP8[((ptr)>>0)]=value; break;
        case 'i16': HEAP16[((ptr)>>1)]=value; break;
        case 'i32': HEAP32[((ptr)>>2)]=value; break;
        case 'i64': (tempI64 = [value>>>0,(tempDouble=value,(+(Math_abs(tempDouble))) >= 1.0 ? (tempDouble > 0.0 ? ((Math_min((+(Math_floor((tempDouble)/4294967296.0))), 4294967295.0))|0)>>>0 : (~~((+(Math_ceil((tempDouble - +(((~~(tempDouble)))>>>0))/4294967296.0)))))>>>0) : 0)],HEAP32[((ptr)>>2)]=tempI64[0],HEAP32[(((ptr)+(4))>>2)]=tempI64[1]); break;
        case 'float': HEAPF32[((ptr)>>2)]=value; break;
        case 'double': HEAPF64[((ptr)>>3)]=value; break;
        default: abort('invalid type for setValue: ' + type);
      }
    } else {
      switch(type) {
        case 'i1': SAFE_HEAP_STORE(((ptr)|0), ((value)|0), 1); break;
        case 'i8': SAFE_HEAP_STORE(((ptr)|0), ((value)|0), 1); break;
        case 'i16': SAFE_HEAP_STORE(((ptr)|0), ((value)|0), 2); break;
        case 'i32': SAFE_HEAP_STORE(((ptr)|0), ((value)|0), 4); break;
        case 'i64': (tempI64 = [value>>>0,(tempDouble=value,(+(Math_abs(tempDouble))) >= 1.0 ? (tempDouble > 0.0 ? ((Math_min((+(Math_floor((tempDouble)/4294967296.0))), 4294967295.0))|0)>>>0 : (~~((+(Math_ceil((tempDouble - +(((~~(tempDouble)))>>>0))/4294967296.0)))))>>>0) : 0)],SAFE_HEAP_STORE(((ptr)|0), ((tempI64[0])|0), 4),SAFE_HEAP_STORE((((ptr)+(4))|0), ((tempI64[1])|0), 4)); break;
        case 'float': SAFE_HEAP_STORE_D(((ptr)|0), Math_fround(value), 4); break;
        case 'double': SAFE_HEAP_STORE_D(((ptr)|0), (+(value)), 8); break;
        default: abort('invalid type for setValue: ' + type);
      }
    }
  }

  /** @type {function(number, string, boolean=)} */
  function getValue(ptr, type, noSafe) {
    type = type || 'i8';
    if (type.charAt(type.length-1) === '*') type = 'i32'; // pointers are 32-bit
    if (noSafe) {
      switch(type) {
        case 'i1': return HEAP8[((ptr)>>0)];
        case 'i8': return HEAP8[((ptr)>>0)];
        case 'i16': return HEAP16[((ptr)>>1)];
        case 'i32': return HEAP32[((ptr)>>2)];
        case 'i64': return HEAP32[((ptr)>>2)];
        case 'float': return HEAPF32[((ptr)>>2)];
        case 'double': return HEAPF64[((ptr)>>3)];
        default: abort('invalid type for getValue: ' + type);
      }
    } else {
      switch(type) {
        case 'i1': return ((SAFE_HEAP_LOAD(((ptr)|0), 1, 0))|0);
        case 'i8': return ((SAFE_HEAP_LOAD(((ptr)|0), 1, 0))|0);
        case 'i16': return ((SAFE_HEAP_LOAD(((ptr)|0), 2, 0))|0);
        case 'i32': return ((SAFE_HEAP_LOAD(((ptr)|0), 4, 0))|0);
        case 'i64': return ((SAFE_HEAP_LOAD(((ptr)|0), 8, 0))|0);
        case 'float': return Math_fround(SAFE_HEAP_LOAD_D(((ptr)|0), 4, 0));
        case 'double': return (+(SAFE_HEAP_LOAD_D(((ptr)|0), 8, 0)));
        default: abort('invalid type for getValue: ' + type);
      }
    }
    return null;
  }


  function getSafeHeapType(bytes, isFloat) {
    switch (bytes) {
      case 1: return 'i8';
      case 2: return 'i16';
      case 4: return isFloat ? 'float' : 'i32';
      case 8: return 'double';
      default: assert(0);
    }
  }

  var SAFE_HEAP_COUNTER = 0;

  function SAFE_HEAP_STORE(dest, value, bytes, isFloat) {
    out('SAFE_HEAP store: ' + [dest, value, bytes, isFloat, SAFE_HEAP_COUNTER++]);
    if (dest <= 0) abort('segmentation fault storing ' + bytes + ' bytes to address ' + dest);
    if (dest % bytes !== 0) abort('alignment error storing to address ' + dest + ', which was expected to be aligned to a multiple of ' + bytes);
    if (dest + bytes > HEAP32[DYNAMICTOP_PTR>>2]) abort('segmentation fault, exceeded the top of the available dynamic heap when storing ' + bytes + ' bytes to address ' + dest + '. DYNAMICTOP=' + HEAP32[DYNAMICTOP_PTR>>2]);
    assert(DYNAMICTOP_PTR);
    assert(HEAP32[DYNAMICTOP_PTR>>2] <= HEAP8.length);
    setValue(dest, value, getSafeHeapType(bytes, isFloat), 1);
  }
  function SAFE_HEAP_STORE_D(dest, value, bytes) {
    SAFE_HEAP_STORE(dest, value, bytes, true);
  }

  function SAFE_HEAP_LOAD(dest, bytes, unsigned, isFloat) {
    if (dest <= 0) abort('segmentation fault loading ' + bytes + ' bytes from address ' + dest);
    if (dest % bytes !== 0) abort('alignment error loading from address ' + dest + ', which was expected to be aligned to a multiple of ' + bytes);
    if (dest + bytes > HEAP32[DYNAMICTOP_PTR>>2]) abort('segmentation fault, exceeded the top of the available dynamic heap when loading ' + bytes + ' bytes from address ' + dest + '. DYNAMICTOP=' + HEAP32[DYNAMICTOP_PTR>>2]);
    assert(DYNAMICTOP_PTR);
    assert(HEAP32[DYNAMICTOP_PTR>>2] <= HEAP8.length);
    var type = getSafeHeapType(bytes, isFloat);
    var ret = getValue(dest, type, 1);
    if (unsigned) ret = unSign(ret, parseInt(type.substr(1)));
    out('SAFE_HEAP load: ' + [dest, ret, bytes, isFloat, unsigned, SAFE_HEAP_COUNTER++]);
    return ret;
  }
  function SAFE_HEAP_LOAD_D(dest, bytes, unsigned) {
    return SAFE_HEAP_LOAD(dest, bytes, unsigned, true);
  }

  function segfault() {
    abort('segmentation fault');
  }
  function alignfault() {
    abort('alignment fault');
  }



  // Wasm globals

  var wasmMemory;

  // In fastcomp asm.js, we don't need a wasm Table at all.
  // In the wasm backend, we polyfill the WebAssembly object,
  // so this creates a (non-native-wasm) table for us.
  var wasmTable = new WebAssembly.Table({
    'initial': 303,
    'maximum': 303 + 0,
    'element': 'anyfunc'
  });


  //========================================
  // Runtime essentials
  //========================================

  // whether we are quitting the application. no code should run after this.
  // set in exit() and abort()
  var ABORT = false;

  /** @type {function(*, string=)} */
  function assert(condition, text) {
    if (!condition) {
      abort('Assertion failed: ' + text);
    }
  }

  // Returns the C function with a specified identifier (for C++, you need to do manual name mangling)
  function getCFunc(ident) {
    var func = Module['_' + ident]; // closure exported function
    assert(func, 'Cannot call unknown function ' + ident + ', make sure it is exported');
    return func;
  }

  // C calling interface.
  function ccall(ident, returnType, argTypes, args, opts) {
    // For fast lookup of conversion functions
    var toC = {
      'string': function(str) {
        var ret = 0;
        if (str !== null && str !== undefined && str !== 0) { // null string
          // at most 4 bytes per UTF-8 code point, +1 for the trailing '\0'
          var len = (str.length << 2) + 1;
          ret = stackAlloc(len);
          stringToUTF8(str, ret, len);
        }
        return ret;
      },
      'array': function(arr) {
        var ret = stackAlloc(arr.length);
        writeArrayToMemory(arr, ret);
        return ret;
      }
    };

    function convertReturnValue(ret) {
      if (returnType === 'string') return UTF8ToString(ret);
      if (returnType === 'boolean') return Boolean(ret);
      return ret;
    }

    var func = getCFunc(ident);
    var cArgs = [];
    var stack = 0;
    assert(returnType !== 'array', 'Return type should not be "array".');
    if (args) {
      for (var i = 0; i < args.length; i++) {
        var converter = toC[argTypes[i]];
        if (converter) {
          if (stack === 0) stack = stackSave();
          cArgs[i] = converter(args[i]);
        } else {
          cArgs[i] = args[i];
        }
      }
    }
    var ret = func.apply(null, cArgs);

    ret = convertReturnValue(ret);
    if (stack !== 0) stackRestore(stack);
    return ret;
  }

  function cwrap(ident, returnType, argTypes, opts) {
    return function() {
      return ccall(ident, returnType, argTypes, arguments);
    }
  }


  // Given a pointer 'ptr' to a null-terminated UTF8-encoded string in the given array that contains uint8 values, returns
  // a copy of that string as a Javascript String object.

  var UTF8Decoder = typeof TextDecoder !== 'undefined' ? new TextDecoder('utf8') : undefined;

  /**
   * @param {number} idx
   * @param {number=} maxBytesToRead
   * @return {string}
   */
  function UTF8ArrayToString(u8Array, idx, maxBytesToRead) {
    var endIdx = idx + maxBytesToRead;
    var endPtr = idx;
    // TextDecoder needs to know the byte length in advance, it doesn't stop on null terminator by itself.
    // Also, use the length info to avoid running tiny strings through TextDecoder, since .subarray() allocates garbage.
    // (As a tiny code save trick, compare endPtr against endIdx using a negation, so that undefined means Infinity)
    while (u8Array[endPtr] && !(endPtr >= endIdx)) ++endPtr;

    if (endPtr - idx > 16 && u8Array.subarray && UTF8Decoder) {
      return UTF8Decoder.decode(u8Array.subarray(idx, endPtr));
    } else {
      var str = '';
      // If building with TextDecoder, we have already computed the string length above, so test loop end condition against that
      while (idx < endPtr) {
        // For UTF8 byte structure, see:
        // http://en.wikipedia.org/wiki/UTF-8#Description
        // https://www.ietf.org/rfc/rfc2279.txt
        // https://tools.ietf.org/html/rfc3629
        var u0 = u8Array[idx++];
        if (!(u0 & 0x80)) { str += String.fromCharCode(u0); continue; }
        var u1 = u8Array[idx++] & 63;
        if ((u0 & 0xE0) == 0xC0) { str += String.fromCharCode(((u0 & 31) << 6) | u1); continue; }
        var u2 = u8Array[idx++] & 63;
        if ((u0 & 0xF0) == 0xE0) {
          u0 = ((u0 & 15) << 12) | (u1 << 6) | u2;
        } else {
          if ((u0 & 0xF8) != 0xF0) warnOnce('Invalid UTF-8 leading byte 0x' + u0.toString(16) + ' encountered when deserializing a UTF-8 string on the asm.js/wasm heap to a JS string!');
          u0 = ((u0 & 7) << 18) | (u1 << 12) | (u2 << 6) | (u8Array[idx++] & 63);
        }

        if (u0 < 0x10000) {
          str += String.fromCharCode(u0);
        } else {
          var ch = u0 - 0x10000;
          str += String.fromCharCode(0xD800 | (ch >> 10), 0xDC00 | (ch & 0x3FF));
        }
      }
    }
    return str;
  }

  // Given a pointer 'ptr' to a null-terminated UTF8-encoded string in the emscripten HEAP, returns a
  // copy of that string as a Javascript String object.
  // maxBytesToRead: an optional length that specifies the maximum number of bytes to read. You can omit
  //                 this parameter to scan the string until the first \0 byte. If maxBytesToRead is
  //                 passed, and the string at [ptr, ptr+maxBytesToReadr[ contains a null byte in the
  //                 middle, then the string will cut short at that byte index (i.e. maxBytesToRead will
  //                 not produce a string of exact length [ptr, ptr+maxBytesToRead[)
  //                 N.B. mixing frequent uses of UTF8ToString() with and without maxBytesToRead may
  //                 throw JS JIT optimizations off, so it is worth to consider consistently using one
  //                 style or the other.
  /**
   * @param {number} ptr
   * @param {number=} maxBytesToRead
   * @return {string}
   */
  function UTF8ToString(ptr, maxBytesToRead) {
    return ptr ? UTF8ArrayToString(HEAPU8, ptr, maxBytesToRead) : '';
  }

  // Copies the given Javascript String object 'str' to the given byte array at address 'outIdx',
  // encoded in UTF8 form and null-terminated. The copy will require at most str.length*4+1 bytes of space in the HEAP.
  // Use the function lengthBytesUTF8 to compute the exact number of bytes (excluding null terminator) that this function will write.
  // Parameters:
  //   str: the Javascript string to copy.
  //   outU8Array: the array to copy to. Each index in this array is assumed to be one 8-byte element.
  //   outIdx: The starting offset in the array to begin the copying.
  //   maxBytesToWrite: The maximum number of bytes this function can write to the array.
  //                    This count should include the null terminator,
  //                    i.e. if maxBytesToWrite=1, only the null terminator will be written and nothing else.
  //                    maxBytesToWrite=0 does not write any bytes to the output, not even the null terminator.
  // Returns the number of bytes written, EXCLUDING the null terminator.

  function stringToUTF8Array(str, outU8Array, outIdx, maxBytesToWrite) {
    if (!(maxBytesToWrite > 0)) // Parameter maxBytesToWrite is not optional. Negative values, 0, null, undefined and false each don't write out any bytes.
      return 0;

    var startIdx = outIdx;
    var endIdx = outIdx + maxBytesToWrite - 1; // -1 for string null terminator.
    for (var i = 0; i < str.length; ++i) {
      // Gotcha: charCodeAt returns a 16-bit word that is a UTF-16 encoded code unit, not a Unicode code point of the character! So decode UTF16->UTF32->UTF8.
      // See http://unicode.org/faq/utf_bom.html#utf16-3
      // For UTF8 byte structure, see http://en.wikipedia.org/wiki/UTF-8#Description and https://www.ietf.org/rfc/rfc2279.txt and https://tools.ietf.org/html/rfc3629
      var u = str.charCodeAt(i); // possibly a lead surrogate
      if (u >= 0xD800 && u <= 0xDFFF) {
        var u1 = str.charCodeAt(++i);
        u = 0x10000 + ((u & 0x3FF) << 10) | (u1 & 0x3FF);
      }
      if (u <= 0x7F) {
        if (outIdx >= endIdx) break;
        outU8Array[outIdx++] = u;
      } else if (u <= 0x7FF) {
        if (outIdx + 1 >= endIdx) break;
        outU8Array[outIdx++] = 0xC0 | (u >> 6);
        outU8Array[outIdx++] = 0x80 | (u & 63);
      } else if (u <= 0xFFFF) {
        if (outIdx + 2 >= endIdx) break;
        outU8Array[outIdx++] = 0xE0 | (u >> 12);
        outU8Array[outIdx++] = 0x80 | ((u >> 6) & 63);
        outU8Array[outIdx++] = 0x80 | (u & 63);
      } else {
        if (outIdx + 3 >= endIdx) break;
        if (u >= 0x200000) warnOnce('Invalid Unicode code point 0x' + u.toString(16) + ' encountered when serializing a JS string to an UTF-8 string on the asm.js/wasm heap! (Valid unicode code points should be in range 0-0x1FFFFF).');
        outU8Array[outIdx++] = 0xF0 | (u >> 18);
        outU8Array[outIdx++] = 0x80 | ((u >> 12) & 63);
        outU8Array[outIdx++] = 0x80 | ((u >> 6) & 63);
        outU8Array[outIdx++] = 0x80 | (u & 63);
      }
    }
    // Null-terminate the pointer to the buffer.
    outU8Array[outIdx] = 0;
    return outIdx - startIdx;
  }

  // Copies the given Javascript String object 'str' to the emscripten HEAP at address 'outPtr',
  // null-terminated and encoded in UTF8 form. The copy will require at most str.length*4+1 bytes of space in the HEAP.
  // Use the function lengthBytesUTF8 to compute the exact number of bytes (excluding null terminator) that this function will write.
  // Returns the number of bytes written, EXCLUDING the null terminator.

  function stringToUTF8(str, outPtr, maxBytesToWrite) {
    assert(typeof maxBytesToWrite == 'number', 'stringToUTF8(str, outPtr, maxBytesToWrite) is missing the third parameter that specifies the length of the output buffer!');
    return stringToUTF8Array(str, HEAPU8,outPtr, maxBytesToWrite);
  }

  // Returns the number of bytes the given Javascript string takes if encoded as a UTF8 byte array, EXCLUDING the null terminator byte.
  function lengthBytesUTF8(str) {
    var len = 0;
    for (var i = 0; i < str.length; ++i) {
      // Gotcha: charCodeAt returns a 16-bit word that is a UTF-16 encoded code unit, not a Unicode code point of the character! So decode UTF16->UTF32->UTF8.
      // See http://unicode.org/faq/utf_bom.html#utf16-3
      var u = str.charCodeAt(i); // possibly a lead surrogate
      if (u >= 0xD800 && u <= 0xDFFF) u = 0x10000 + ((u & 0x3FF) << 10) | (str.charCodeAt(++i) & 0x3FF);
      if (u <= 0x7F) ++len;
      else if (u <= 0x7FF) len += 2;
      else if (u <= 0xFFFF) len += 3;
      else len += 4;
    }
    return len;
  }


  // Given a pointer 'ptr' to a null-terminated UTF16LE-encoded string in the emscripten HEAP, returns
  // a copy of that string as a Javascript String object.

  var UTF16Decoder = typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-16le') : undefined;

  function writeArrayToMemory(array, buffer) {
    assert(array.length >= 0, 'writeArrayToMemory array must have a length (should be an array or typed array)');
    HEAP8.set(array, buffer);
  }
  var WASM_PAGE_SIZE = 65536;

  var /** @type {ArrayBuffer} */
    buffer,
  /** @type {Int8Array} */
    HEAP8,
  /** @type {Uint8Array} */
    HEAPU8,
  /** @type {Int16Array} */
    HEAP16,
  /** @type {Uint16Array} */
    HEAPU16,
  /** @type {Int32Array} */
    HEAP32,
  /** @type {Uint32Array} */
    HEAPU32,
  /** @type {Float32Array} */
    HEAPF32,
  /** @type {Float64Array} */
    HEAPF64;

  function updateGlobalBufferAndViews(buf) {
    buffer = buf;
    Module['HEAP8'] = HEAP8 = new Int8Array(buf);
    Module['HEAP16'] = HEAP16 = new Int16Array(buf);
    Module['HEAP32'] = HEAP32 = new Int32Array(buf);
    Module['HEAPU8'] = HEAPU8 = new Uint8Array(buf);
    Module['HEAPU16'] = HEAPU16 = new Uint16Array(buf);
    Module['HEAPU32'] = HEAPU32 = new Uint32Array(buf);
    Module['HEAPF32'] = HEAPF32 = new Float32Array(buf);
    Module['HEAPF64'] = HEAPF64 = new Float64Array(buf);
  }

  var STACK_BASE = 5270032,
      STACK_MAX = 27152,
      DYNAMIC_BASE = 5270032,
      DYNAMICTOP_PTR = 26976;

  assert(STACK_BASE % 16 === 0, 'stack must start aligned');
  assert(DYNAMIC_BASE % 16 === 0, 'heap must start aligned');



  var TOTAL_STACK = 5242880;
  if (Module['TOTAL_STACK']) assert(TOTAL_STACK === Module['TOTAL_STACK'], 'the stack size can no longer be determined at runtime');

  var INITIAL_TOTAL_MEMORY = Module['TOTAL_MEMORY'] || 536870912;if (!Object.getOwnPropertyDescriptor(Module, 'TOTAL_MEMORY')) Object.defineProperty(Module, 'TOTAL_MEMORY', { configurable: true, get: function() { abort('Module.TOTAL_MEMORY has been replaced with plain INITIAL_TOTAL_MEMORY'); } });

  assert(INITIAL_TOTAL_MEMORY >= TOTAL_STACK, 'TOTAL_MEMORY should be larger than TOTAL_STACK, was ' + INITIAL_TOTAL_MEMORY + '! (TOTAL_STACK=' + TOTAL_STACK + ')');

  // check for full engine support (use string 'subarray' to avoid closure compiler confusion)
  assert(typeof Int32Array !== 'undefined' && typeof Float64Array !== 'undefined' && Int32Array.prototype.subarray !== undefined && Int32Array.prototype.set !== undefined,
         'JS engine does not provide full typed array support');






  // In standalone mode, the wasm creates the memory, and the user can't provide it.
  // In non-standalone/normal mode, we create the memory here.

  // Create the main memory. (Note: this isn't used in STANDALONE_WASM mode since the wasm
  // memory is created in the wasm, not in JS.)

    if (Module['wasmMemory']) {
      wasmMemory = Module['wasmMemory'];
    } else
    {
      wasmMemory = new WebAssembly.Memory({
        'initial': INITIAL_TOTAL_MEMORY / WASM_PAGE_SIZE
        ,
        'maximum': INITIAL_TOTAL_MEMORY / WASM_PAGE_SIZE
      });
    }


  if (wasmMemory) {
    buffer = wasmMemory.buffer;
  }

  // If the user provides an incorrect length, just use that length instead rather than providing the user to
  // specifically provide the memory length with Module['TOTAL_MEMORY'].
  INITIAL_TOTAL_MEMORY = buffer.byteLength;
  assert(INITIAL_TOTAL_MEMORY % WASM_PAGE_SIZE === 0);
  updateGlobalBufferAndViews(buffer);

  HEAP32[DYNAMICTOP_PTR>>2] = DYNAMIC_BASE;




  // Initializes the stack cookie. Called at the startup of main and at the startup of each thread in pthreads mode.
  function writeStackCookie() {
    assert((STACK_MAX & 3) == 0);
    // The stack grows downwards
    HEAPU32[(STACK_MAX >> 2)+1] = 0x02135467;
    HEAPU32[(STACK_MAX >> 2)+2] = 0x89BACDFE;
    // Also test the global address 0 for integrity.
    // We don't do this with ASan because ASan does its own checks for this.
    HEAP32[0] = 0x63736d65; /* 'emsc' */
  }

  function checkStackCookie() {
    var cookie1 = HEAPU32[(STACK_MAX >> 2)+1];
    var cookie2 = HEAPU32[(STACK_MAX >> 2)+2];
    if (cookie1 != 0x02135467 || cookie2 != 0x89BACDFE) {
      abort('Stack overflow! Stack cookie has been overwritten, expected hex dwords 0x89BACDFE and 0x02135467, but received 0x' + cookie2.toString(16) + ' ' + cookie1.toString(16));
    }
    // Also test the global address 0 for integrity.
    // We don't do this with ASan because ASan does its own checks for this.
    if (HEAP32[0] !== 0x63736d65 /* 'emsc' */) abort('Runtime error: The application has corrupted its heap memory area (address zero)!');
  }

  function abortStackOverflow(allocSize) {
    abort('Stack overflow! Attempted to allocate ' + allocSize + ' bytes on the stack, but stack has only ' + (STACK_MAX - stackSave() + allocSize) + ' bytes available!');
  }




  // Endianness check (note: assumes compiler arch was little-endian)
  (function() {
    var h16 = new Int16Array(1);
    var h8 = new Int8Array(h16.buffer);
    h16[0] = 0x6373;
    if (h8[0] !== 0x73 || h8[1] !== 0x63) throw 'Runtime error: expected the system to be little-endian!';
  })();



  function callRuntimeCallbacks(callbacks) {
    while(callbacks.length > 0) {
      var callback = callbacks.shift();
      if (typeof callback == 'function') {
        callback();
        continue;
      }
      var func = callback.func;
      if (typeof func === 'number') {
        if (callback.arg === undefined) {
          Module['dynCall_v'](func);
        } else {
          Module['dynCall_vi'](func, callback.arg);
        }
      } else {
        func(callback.arg === undefined ? null : callback.arg);
      }
    }
  }

  var __ATPRERUN__  = []; // functions called before the runtime is initialized
  var __ATINIT__    = []; // functions called during startup
  var __ATMAIN__    = []; // functions called when main() is to be run
  var __ATPOSTRUN__ = []; // functions called after the main() is called

  var runtimeInitialized = false;
  var runtimeExited = false;


  function preRun() {

    if (Module['preRun']) {
      if (typeof Module['preRun'] == 'function') Module['preRun'] = [Module['preRun']];
      while (Module['preRun'].length) {
        addOnPreRun(Module['preRun'].shift());
      }
    }

    callRuntimeCallbacks(__ATPRERUN__);
  }

  function initRuntime() {
    checkStackCookie();
    assert(!runtimeInitialized);
    runtimeInitialized = true;
    
    callRuntimeCallbacks(__ATINIT__);
  }

  function preMain() {
    checkStackCookie();
    
    callRuntimeCallbacks(__ATMAIN__);
  }

  function postRun() {
    checkStackCookie();

    if (Module['postRun']) {
      if (typeof Module['postRun'] == 'function') Module['postRun'] = [Module['postRun']];
      while (Module['postRun'].length) {
        addOnPostRun(Module['postRun'].shift());
      }
    }

    callRuntimeCallbacks(__ATPOSTRUN__);
  }

  function addOnPreRun(cb) {
    __ATPRERUN__.unshift(cb);
  }

  function addOnPostRun(cb) {
    __ATPOSTRUN__.unshift(cb);
  }

  function unSign(value, bits, ignore) {
    if (value >= 0) {
      return value;
    }
    return bits <= 32 ? 2*Math.abs(1 << (bits-1)) + value // Need some trickery, since if bits == 32, we are right at the limit of the bits JS uses in bitshifts
                      : Math.pow(2, bits)         + value;
  }


  assert(Math.imul, 'This browser does not support Math.imul(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill');
  assert(Math.fround, 'This browser does not support Math.fround(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill');
  assert(Math.clz32, 'This browser does not support Math.clz32(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill');
  assert(Math.trunc, 'This browser does not support Math.trunc(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill');

  var Math_abs = Math.abs;
  var Math_ceil = Math.ceil;
  var Math_floor = Math.floor;
  var Math_fround = Math.fround;
  var Math_min = Math.min;



  // A counter of dependencies for calling run(). If we need to
  // do asynchronous work before running, increment this and
  // decrement it. Incrementing must happen in a place like
  // Module.preRun (used by emcc to add file preloading).
  // Note that you can add dependencies in preRun, even though
  // it happens right before run - run will be postponed until
  // the dependencies are met.
  var runDependencies = 0;
  var runDependencyWatcher = null;
  var dependenciesFulfilled = null; // overridden to take different actions when all run dependencies are fulfilled
  var runDependencyTracking = {};

  function addRunDependency(id) {
    runDependencies++;

    if (Module['monitorRunDependencies']) {
      Module['monitorRunDependencies'](runDependencies);
    }

    if (id) {
      assert(!runDependencyTracking[id]);
      runDependencyTracking[id] = 1;
      if (runDependencyWatcher === null && typeof setInterval !== 'undefined') {
        // Check for missing dependencies every few seconds
        runDependencyWatcher = setInterval(function() {
          if (ABORT) {
            clearInterval(runDependencyWatcher);
            runDependencyWatcher = null;
            return;
          }
          var shown = false;
          for (var dep in runDependencyTracking) {
            if (!shown) {
              shown = true;
              err('still waiting on run dependencies:');
            }
            err('dependency: ' + dep);
          }
          if (shown) {
            err('(end of list)');
          }
        }, 10000);
      }
    } else {
      err('warning: run dependency added without ID');
    }
  }

  function removeRunDependency(id) {
    runDependencies--;

    if (Module['monitorRunDependencies']) {
      Module['monitorRunDependencies'](runDependencies);
    }

    if (id) {
      assert(runDependencyTracking[id]);
      delete runDependencyTracking[id];
    } else {
      err('warning: run dependency removed without ID');
    }
    if (runDependencies == 0) {
      if (runDependencyWatcher !== null) {
        clearInterval(runDependencyWatcher);
        runDependencyWatcher = null;
      }
      if (dependenciesFulfilled) {
        var callback = dependenciesFulfilled;
        dependenciesFulfilled = null;
        callback(); // can add another dependenciesFulfilled
      }
    }
  }

  Module["preloadedImages"] = {}; // maps url to image data
  Module["preloadedAudios"] = {}; // maps url to audio data


  function abort(what) {
    if (Module['onAbort']) {
      Module['onAbort'](what);
    }

    what += '';
    out(what);
    err(what);

    ABORT = true;

    var output = 'abort(' + what + ') at ' + stackTrace();
    what = output;

    // Throw a wasm runtime error, because a JS error might be seen as a foreign
    // exception, which means we'd run destructors on it. We need the error to
    // simply make the program stop.
    throw new WebAssembly.RuntimeError(what);
  }




  // show errors on likely calls to FS when it was not included
  var FS = {
    error: function() {
      abort('Filesystem support (FS) was not included. The problem is that you are using files from JS, but files were not used from C/C++, so filesystem support was not auto-included. You can force-include filesystem support with  -s FORCE_FILESYSTEM=1');
    },
    init: function() { FS.error(); },
    createDataFile: function() { FS.error(); },
    createPreloadedFile: function() { FS.error(); },
    createLazyFile: function() { FS.error(); },
    open: function() { FS.error(); },
    mkdev: function() { FS.error(); },
    registerDevice: function() { FS.error(); },
    analyzePath: function() { FS.error(); },
    loadFilesFromDB: function() { FS.error(); },

    ErrnoError: function ErrnoError() { FS.error(); },
  };
  Module['FS_createDataFile'] = FS.createDataFile;
  Module['FS_createPreloadedFile'] = FS.createPreloadedFile;



  // Copyright 2017 The Emscripten Authors.  All rights reserved.
  // Emscripten is available under two separate licenses, the MIT license and the
  // University of Illinois/NCSA Open Source License.  Both these licenses can be
  // found in the LICENSE file.

  // Prefix of data URIs emitted by SINGLE_FILE and related options.
  var dataURIPrefix = 'data:application/octet-stream;base64,';

  // Indicates whether filename is a base64 data URI.
  function isDataURI(filename) {
    return String.prototype.startsWith ?
        filename.startsWith(dataURIPrefix) :
        filename.indexOf(dataURIPrefix) === 0;
  }




  var wasmBinaryFile = 'tfjs-backend-wasm.wasm';
  if (!isDataURI(wasmBinaryFile)) {
    wasmBinaryFile = locateFile(wasmBinaryFile);
  }

  function getBinary() {
    try {
      if (wasmBinary) {
        return new Uint8Array(wasmBinary);
      }

      if (readBinary) {
        return readBinary(wasmBinaryFile);
      } else {
        throw "both async and sync fetching of the wasm failed";
      }
    }
    catch (err) {
      abort(err);
    }
  }

  function getBinaryPromise() {
    // if we don't have the binary yet, and have the Fetch api, use that
    // in some environments, like Electron's render process, Fetch api may be present, but have a different context than expected, let's only use it on the Web
    if (!wasmBinary && (ENVIRONMENT_IS_WEB ) && typeof fetch === 'function') {
      return fetch(wasmBinaryFile, { credentials: 'same-origin' }).then(function(response) {
        if (!response['ok']) {
          throw "failed to load wasm binary file at '" + wasmBinaryFile + "'";
        }
        return response['arrayBuffer']();
      }).catch(function () {
        return getBinary();
      });
    }
    // Otherwise, getBinary should be able to get it synchronously
    return new Promise(function(resolve, reject) {
      resolve(getBinary());
    });
  }



  // Create the wasm instance.
  // Receives the wasm imports, returns the exports.
  function createWasm() {
    // prepare imports
    var info = {
      'env': asmLibraryArg,
      'wasi_unstable': asmLibraryArg
    };
    // Load the wasm module and create an instance of using native support in the JS engine.
    // handle a generated wasm instance, receiving its exports and
    // performing other necessary setup
    function receiveInstance(instance, module) {
      var exports = instance.exports;
      Module['asm'] = exports;
      removeRunDependency('wasm-instantiate');
    }
     // we can't run yet (except in a pthread, where we have a custom sync instantiator)
    addRunDependency('wasm-instantiate');


    // Async compilation can be confusing when an error on the page overwrites Module
    // (for example, if the order of elements is wrong, and the one defining Module is
    // later), so we save Module and check it later.
    var trueModule = Module;
    function receiveInstantiatedSource(output) {
      // 'output' is a WebAssemblyInstantiatedSource object which has both the module and instance.
      // receiveInstance() will swap in the exports (to Module.asm) so they can be called
      assert(Module === trueModule, 'the Module object should not be replaced during async compilation - perhaps the order of HTML elements is wrong?');
      trueModule = null;
        // TODO: Due to Closure regression https://github.com/google/closure-compiler/issues/3193, the above line no longer optimizes out down to the following line.
        // When the regression is fixed, can restore the above USE_PTHREADS-enabled path.
      receiveInstance(output['instance']);
    }


    function instantiateArrayBuffer(receiver) {
      return getBinaryPromise().then(function(binary) {
        return WebAssembly.instantiate(binary, info);
      }).then(receiver, function(reason) {
        err('failed to asynchronously prepare wasm: ' + reason);
        abort(reason);
      });
    }

    // Prefer streaming instantiation if available.
    function instantiateAsync() {
      if (!wasmBinary &&
          typeof WebAssembly.instantiateStreaming === 'function' &&
          !isDataURI(wasmBinaryFile) &&
          typeof fetch === 'function') {
        fetch(wasmBinaryFile, { credentials: 'same-origin' }).then(function (response) {
          var result = WebAssembly.instantiateStreaming(response, info);
          return result.then(receiveInstantiatedSource, function(reason) {
              // We expect the most common failure cause to be a bad MIME type for the binary,
              // in which case falling back to ArrayBuffer instantiation should work.
              err('wasm streaming compile failed: ' + reason);
              err('falling back to ArrayBuffer instantiation');
              instantiateArrayBuffer(receiveInstantiatedSource);
            });
        });
      } else {
        return instantiateArrayBuffer(receiveInstantiatedSource);
      }
    }
    // User shell pages can write their own Module.instantiateWasm = function(imports, successCallback) callback
    // to manually instantiate the Wasm module themselves. This allows pages to run the instantiation parallel
    // to any other async startup actions they are performing.
    if (Module['instantiateWasm']) {
      try {
        var exports = Module['instantiateWasm'](info, receiveInstance);
        return exports;
      } catch(e) {
        err('Module.instantiateWasm callback failed with error: ' + e);
        return false;
      }
    }

    instantiateAsync();
    return {}; // no exports yet; we'll fill them in later
  }


  // Globals used by JS i64 conversions
  var tempDouble;
  var tempI64;




  // STATICTOP = STATIC_BASE + 26128;
  /* global initializers */  __ATINIT__.push({ func: function() { ___wasm_call_ctors(); } });



  /* no memory initializer */
  // {{PRE_LIBRARY}}


    function demangle(func) {
        var __cxa_demangle_func = Module['___cxa_demangle'] || Module['__cxa_demangle'];
        assert(__cxa_demangle_func);
        try {
          var s = func;
          if (s.startsWith('__Z'))
            s = s.substr(1);
          var len = lengthBytesUTF8(s)+1;
          var buf = _malloc(len);
          stringToUTF8(s, buf, len);
          var status = _malloc(4);
          var ret = __cxa_demangle_func(buf, 0, 0, status);
          if (((SAFE_HEAP_LOAD(((status)|0), 4, 0))|0) === 0 && ret) {
            return UTF8ToString(ret);
          }
          // otherwise, libcxxabi failed
        } catch(e) {
          // ignore problems here
        } finally {
          if (buf) _free(buf);
          if (status) _free(status);
          if (ret) _free(ret);
        }
        // failure when using libcxxabi, don't demangle
        return func;
      }

    function demangleAll(text) {
        var regex =
          /\b_Z[\w\d_]+/g;
        return text.replace(regex,
          function(x) {
            var y = demangle(x);
            return x === y ? x : (y + ' [' + x + ']');
          });
      }

    function jsStackTrace() {
        var err = new Error();
        if (!err.stack) {
          // IE10+ special cases: It does have callstack info, but it is only populated if an Error object is thrown,
          // so try that as a special-case.
          try {
            throw new Error(0);
          } catch(e) {
            err = e;
          }
          if (!err.stack) {
            return '(no stack trace available)';
          }
        }
        return err.stack.toString();
      }

    function stackTrace() {
        var js = jsStackTrace();
        if (Module['extraStackTrace']) js += '\n' + Module['extraStackTrace']();
        return demangleAll(js);
      }

    function ___assert_fail(condition, filename, line, func) {
        abort('Assertion failed: ' + UTF8ToString(condition) + ', at: ' + [filename ? UTF8ToString(filename) : 'unknown filename', line, func ? UTF8ToString(func) : 'unknown function']);
      }

    function ___cxa_allocate_exception(size) {
        return _malloc(size);
      }

    
    function _atexit(func, arg) {
        warnOnce('atexit() called, but EXIT_RUNTIME is not set, so atexits() will not be called. set EXIT_RUNTIME to 1 (see the FAQ)');
      }function ___cxa_atexit(
    ) {
    return _atexit.apply(null, arguments)
    }

    
    var ___exception_infos={};
    
    var ___exception_caught= [];
    
    function ___exception_addRef(ptr) {
        err('addref ' + ptr);
        if (!ptr) return;
        var info = ___exception_infos[ptr];
        info.refcount++;
      }
    
    function ___exception_deAdjust(adjusted) {
        if (!adjusted || ___exception_infos[adjusted]) return adjusted;
        for (var key in ___exception_infos) {
          var ptr = +key; // the iteration key is a string, and if we throw this, it must be an integer as that is what we look for
          var adj = ___exception_infos[ptr].adjusted;
          var len = adj.length;
          for (var i = 0; i < len; i++) {
            if (adj[i] === adjusted) {
              err('de-adjusted exception ptr ' + adjusted + ' to ' + ptr);
              return ptr;
            }
          }
        }
        err('no de-adjustment for unknown exception ptr ' + adjusted);
        return adjusted;
      }
  function ___cxa_begin_catch(ptr) {
        var info = ___exception_infos[ptr];
        if (info && !info.caught) {
          info.caught = true;
        }
        if (info) info.rethrown = false;
        ___exception_caught.push(ptr);
        err('cxa_begin_catch ' + [ptr, 'stack', ___exception_caught]);
        ___exception_addRef(___exception_deAdjust(ptr));
        return ptr;
      }

    
    var ___exception_last=0;
    
    
    function ___cxa_free_exception(ptr) {
        try {
          return _free(ptr);
        } catch(e) {
          err('exception during cxa_free_exception: ' + e);
        }
      }function ___exception_decRef(ptr) {
        err('decref ' + ptr);
        if (!ptr) return;
        var info = ___exception_infos[ptr];
        assert(info.refcount > 0);
        info.refcount--;
        // A rethrown exception can reach refcount 0; it must not be discarded
        // Its next handler will clear the rethrown flag and addRef it, prior to
        // final decRef and destruction here
        if (info.refcount === 0 && !info.rethrown) {
          if (info.destructor) {
            // In Wasm, destructors return 'this' as in ARM
            Module['dynCall_ii'](info.destructor, ptr);
          }
          delete ___exception_infos[ptr];
          ___cxa_free_exception(ptr);
          err('decref freeing exception ' + [ptr, ___exception_last, 'stack', ___exception_caught]);
        }
      }function ___cxa_end_catch() {
        // Clear state flag.
        _setThrew(0);
        // Call destructor if one is registered then clear it.
        var ptr = ___exception_caught.pop();
        err('cxa_end_catch popped ' + [ptr, ___exception_last, 'stack', ___exception_caught]);
        if (ptr) {
          ___exception_decRef(___exception_deAdjust(ptr));
          ___exception_last = 0; // XXX in decRef?
        }
      }

    function ___cxa_find_matching_catch_2() {
        var thrown = ___exception_last;
        if (!thrown) {
          // just pass through the null ptr
          return ((setTempRet0(0),0)|0);
        }
        var info = ___exception_infos[thrown];
        var throwntype = info.type;
        if (!throwntype) {
          // just pass through the thrown ptr
          return ((setTempRet0(0),thrown)|0);
        }
        var typeArray = Array.prototype.slice.call(arguments);
    
        var pointer = ___cxa_is_pointer_type(throwntype);
        // can_catch receives a **, add indirection
        out("can_catch on " + [thrown]);
        var buffer = 27136;
        SAFE_HEAP_STORE(((buffer)|0), ((thrown)|0), 4);
        thrown = buffer;
        // The different catch blocks are denoted by different types.
        // Due to inheritance, those types may not precisely match the
        // type of the thrown object. Find one which matches, and
        // return the type of the catch block which should be called.
        for (var i = 0; i < typeArray.length; i++) {
          if (typeArray[i] && ___cxa_can_catch(typeArray[i], throwntype, thrown)) {
            thrown = ((SAFE_HEAP_LOAD(((thrown)|0), 4, 0))|0); // undo indirection
            info.adjusted.push(thrown);
            out("  can_catch found " + [thrown, typeArray[i]]);
            return ((setTempRet0(typeArray[i]),thrown)|0);
          }
        }
        // Shouldn't happen unless we have bogus data in typeArray
        // or encounter a type for which emscripten doesn't have suitable
        // typeinfo defined. Best-efforts match just in case.
        thrown = ((SAFE_HEAP_LOAD(((thrown)|0), 4, 0))|0); // undo indirection
        return ((setTempRet0(throwntype),thrown)|0);
      }

    function ___cxa_find_matching_catch_3() {
        var thrown = ___exception_last;
        if (!thrown) {
          // just pass through the null ptr
          return ((setTempRet0(0),0)|0);
        }
        var info = ___exception_infos[thrown];
        var throwntype = info.type;
        if (!throwntype) {
          // just pass through the thrown ptr
          return ((setTempRet0(0),thrown)|0);
        }
        var typeArray = Array.prototype.slice.call(arguments);
    
        var pointer = ___cxa_is_pointer_type(throwntype);
        // can_catch receives a **, add indirection
        out("can_catch on " + [thrown]);
        var buffer = 27136;
        SAFE_HEAP_STORE(((buffer)|0), ((thrown)|0), 4);
        thrown = buffer;
        // The different catch blocks are denoted by different types.
        // Due to inheritance, those types may not precisely match the
        // type of the thrown object. Find one which matches, and
        // return the type of the catch block which should be called.
        for (var i = 0; i < typeArray.length; i++) {
          if (typeArray[i] && ___cxa_can_catch(typeArray[i], throwntype, thrown)) {
            thrown = ((SAFE_HEAP_LOAD(((thrown)|0), 4, 0))|0); // undo indirection
            info.adjusted.push(thrown);
            out("  can_catch found " + [thrown, typeArray[i]]);
            return ((setTempRet0(typeArray[i]),thrown)|0);
          }
        }
        // Shouldn't happen unless we have bogus data in typeArray
        // or encounter a type for which emscripten doesn't have suitable
        // typeinfo defined. Best-efforts match just in case.
        thrown = ((SAFE_HEAP_LOAD(((thrown)|0), 4, 0))|0); // undo indirection
        return ((setTempRet0(throwntype),thrown)|0);
      }


    function ___cxa_throw(ptr, type, destructor) {
        err('Compiled code throwing an exception, ' + [ptr,type,destructor]);
        ___exception_infos[ptr] = {
          ptr: ptr,
          adjusted: [ptr],
          type: type,
          destructor: destructor,
          refcount: 0,
          caught: false,
          rethrown: false
        };
        ___exception_last = ptr;
        throw ptr;
      }

    function ___resumeException(ptr) {
        out("Resuming exception " + [ptr, ___exception_last]);
        if (!___exception_last) { ___exception_last = ptr; }
        throw ptr;
      }
  var SYSCALLS={buffers:[null,[],[]],printChar:function(stream, curr) {
          var buffer = SYSCALLS.buffers[stream];
          assert(buffer);
          if (curr === 0 || curr === 10) {
            (stream === 1 ? out : err)(UTF8ArrayToString(buffer, 0));
            buffer.length = 0;
          } else {
            buffer.push(curr);
          }
        },varargs:0,get:function(varargs) {
          SYSCALLS.varargs += 4;
          var ret = ((SAFE_HEAP_LOAD((((SYSCALLS.varargs)-(4))|0), 4, 0))|0);
          return ret;
        },getStr:function() {
          var ret = UTF8ToString(SYSCALLS.get());
          return ret;
        },get64:function() {
          var low = SYSCALLS.get(), high = SYSCALLS.get();
          if (low >= 0) assert(high === 0);
          else assert(high === -1);
          return low;
        },getZero:function() {
          assert(SYSCALLS.get() === 0);
        }};function ___syscall4(which, varargs) {SYSCALLS.varargs = varargs;
    try {
     // write
        // hack to support printf in SYSCALLS_REQUIRE_FILESYSTEM=0
        var stream = SYSCALLS.get(), buf = SYSCALLS.get(), count = SYSCALLS.get();
        for (var i = 0; i < count; i++) {
          SYSCALLS.printChar(stream, HEAPU8[buf+i]);
        }
        return count;
      } catch (e) {
      if (typeof FS === 'undefined' || !(e instanceof FS.ErrnoError)) abort(e);
      return -e.errno;
    }
    }

    function _abort() {
        abort();
      }

    function _emscripten_get_sbrk_ptr() {
        return 26976;
      }

    function _emscripten_memcpy_big(dest, src, num) {
        HEAPU8.set(HEAPU8.subarray(src, src+num), dest);
      }
    
    function abortOnCannotGrowMemory(requestedSize) {
        abort('Cannot enlarge memory arrays to size ' + requestedSize + ' bytes (OOM). Either (1) compile with  -s TOTAL_MEMORY=X  with X higher than the current value ' + HEAP8.length + ', (2) compile with  -s ALLOW_MEMORY_GROWTH=1  which allows increasing the size at runtime, or (3) if you want malloc to return NULL (0) instead of this abort, compile with  -s ABORTING_MALLOC=0 ');
      }function _emscripten_resize_heap(requestedSize) {
        abortOnCannotGrowMemory(requestedSize);
      }

    function _fd_close(fd) {try {
    
        abort('it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM');
        return 0;
      } catch (e) {
      if (typeof FS === 'undefined' || !(e instanceof FS.ErrnoError)) abort(e);
      return e.errno;
    }
    }

    function _fd_seek(fd, offset_low, offset_high, whence, newOffset) {try {
    
        abort('it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM');
        return 0;
      } catch (e) {
      if (typeof FS === 'undefined' || !(e instanceof FS.ErrnoError)) abort(e);
      return e.errno;
    }
    }
  function _fd_write(fd, iov, iovcnt, pnum) {try {
    
        // hack to support printf in SYSCALLS_REQUIRE_FILESYSTEM=0
        var num = 0;
        for (var i = 0; i < iovcnt; i++) {
          var ptr = ((SAFE_HEAP_LOAD((((iov)+(i*8))|0), 4, 0))|0);
          var len = ((SAFE_HEAP_LOAD((((iov)+(i*8 + 4))|0), 4, 0))|0);
          for (var j = 0; j < len; j++) {
            SYSCALLS.printChar(fd, HEAPU8[ptr+j]);
          }
          num += len;
        }
        SAFE_HEAP_STORE(((pnum)|0), ((num)|0), 4);
        return 0;
      } catch (e) {
      if (typeof FS === 'undefined' || !(e instanceof FS.ErrnoError)) abort(e);
      return e.errno;
    }
    }

    function _getTempRet0() {
        return (getTempRet0() | 0);
      }

    
    function _roundf(d) {
        d = +d;
        return d >= +0 ? +Math_floor(d + +0.5) : +Math_ceil(d - +0.5);
      }

    function _setTempRet0($i) {
        setTempRet0(($i) | 0);
      }
  var asmLibraryArg = { "__assert_fail": ___assert_fail, "__cxa_allocate_exception": ___cxa_allocate_exception, "__cxa_atexit": ___cxa_atexit, "__cxa_begin_catch": ___cxa_begin_catch, "__cxa_end_catch": ___cxa_end_catch, "__cxa_find_matching_catch_2": ___cxa_find_matching_catch_2, "__cxa_find_matching_catch_3": ___cxa_find_matching_catch_3, "__cxa_free_exception": ___cxa_free_exception, "__cxa_throw": ___cxa_throw, "__resumeException": ___resumeException, "__syscall4": ___syscall4, "abort": _abort, "alignfault": alignfault, "emscripten_get_sbrk_ptr": _emscripten_get_sbrk_ptr, "emscripten_memcpy_big": _emscripten_memcpy_big, "emscripten_resize_heap": _emscripten_resize_heap, "fd_close": _fd_close, "fd_seek": _fd_seek, "fd_write": _fd_write, "getTempRet0": _getTempRet0, "invoke_i": invoke_i, "invoke_ii": invoke_ii, "invoke_iii": invoke_iii, "invoke_v": invoke_v, "invoke_vii": invoke_vii, "invoke_viii": invoke_viii, "memory": wasmMemory, "roundf": _roundf, "segfault": segfault, "setTempRet0": _setTempRet0, "table": wasmTable };
  var asm = createWasm();
  var real____wasm_call_ctors = asm["__wasm_call_ctors"];
  asm["__wasm_call_ctors"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real____wasm_call_ctors.apply(null, arguments);
  };

  var real__init = asm["init"];
  asm["init"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__init.apply(null, arguments);
  };

  var real__register_tensor = asm["register_tensor"];
  asm["register_tensor"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__register_tensor.apply(null, arguments);
  };

  var real__dispose_data = asm["dispose_data"];
  asm["dispose_data"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__dispose_data.apply(null, arguments);
  };

  var real__dispose = asm["dispose"];
  asm["dispose"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__dispose.apply(null, arguments);
  };

  var real__Abs = asm["Abs"];
  asm["Abs"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__Abs.apply(null, arguments);
  };

  var real__Add = asm["Add"];
  asm["Add"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__Add.apply(null, arguments);
  };

  var real__AddN = asm["AddN"];
  asm["AddN"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__AddN.apply(null, arguments);
  };

  var real__AvgPool = asm["AvgPool"];
  asm["AvgPool"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__AvgPool.apply(null, arguments);
  };

  var real__BatchMatMul = asm["BatchMatMul"];
  asm["BatchMatMul"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__BatchMatMul.apply(null, arguments);
  };

  var real__ClipByValue = asm["ClipByValue"];
  asm["ClipByValue"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__ClipByValue.apply(null, arguments);
  };

  var real__Conv2D = asm["Conv2D"];
  asm["Conv2D"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__Conv2D.apply(null, arguments);
  };

  var real__CropAndResize = asm["CropAndResize"];
  asm["CropAndResize"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__CropAndResize.apply(null, arguments);
  };

  var real__DepthwiseConv2dNative = asm["DepthwiseConv2dNative"];
  asm["DepthwiseConv2dNative"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__DepthwiseConv2dNative.apply(null, arguments);
  };

  var real__Div = asm["Div"];
  asm["Div"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__Div.apply(null, arguments);
  };

  var real__FusedBatchNorm = asm["FusedBatchNorm"];
  asm["FusedBatchNorm"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__FusedBatchNorm.apply(null, arguments);
  };

  var real__FusedConv2D = asm["FusedConv2D"];
  asm["FusedConv2D"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__FusedConv2D.apply(null, arguments);
  };

  var real__Max = asm["Max"];
  asm["Max"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__Max.apply(null, arguments);
  };

  var real__MaxPool = asm["MaxPool"];
  asm["MaxPool"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__MaxPool.apply(null, arguments);
  };

  var real__Min = asm["Min"];
  asm["Min"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__Min.apply(null, arguments);
  };

  var real__Mul = asm["Mul"];
  asm["Mul"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__Mul.apply(null, arguments);
  };

  var real__PadV2 = asm["PadV2"];
  asm["PadV2"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__PadV2.apply(null, arguments);
  };

  var real__Prelu = asm["Prelu"];
  asm["Prelu"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__Prelu.apply(null, arguments);
  };

  var real__Sigmoid = asm["Sigmoid"];
  asm["Sigmoid"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__Sigmoid.apply(null, arguments);
  };

  var real__Square = asm["Square"];
  asm["Square"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__Square.apply(null, arguments);
  };

  var real__Sub = asm["Sub"];
  asm["Sub"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__Sub.apply(null, arguments);
  };

  var real__Transpose = asm["Transpose"];
  asm["Transpose"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__Transpose.apply(null, arguments);
  };

  var real__malloc = asm["malloc"];
  asm["malloc"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__malloc.apply(null, arguments);
  };

  var real__free = asm["free"];
  asm["free"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__free.apply(null, arguments);
  };

  var real__setThrew = asm["setThrew"];
  asm["setThrew"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real__setThrew.apply(null, arguments);
  };

  var real____cxa_demangle = asm["__cxa_demangle"];
  asm["__cxa_demangle"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real____cxa_demangle.apply(null, arguments);
  };

  var real_dynCall_i = asm["dynCall_i"];
  asm["dynCall_i"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_i.apply(null, arguments);
  };

  var real_dynCall_ii = asm["dynCall_ii"];
  asm["dynCall_ii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_ii.apply(null, arguments);
  };

  var real_dynCall_iii = asm["dynCall_iii"];
  asm["dynCall_iii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_iii.apply(null, arguments);
  };

  var real_dynCall_v = asm["dynCall_v"];
  asm["dynCall_v"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_v.apply(null, arguments);
  };

  var real_dynCall_vii = asm["dynCall_vii"];
  asm["dynCall_vii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_vii.apply(null, arguments);
  };

  var real_dynCall_viii = asm["dynCall_viii"];
  asm["dynCall_viii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_viii.apply(null, arguments);
  };

  var real_stackSave = asm["stackSave"];
  asm["stackSave"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_stackSave.apply(null, arguments);
  };

  var real_stackAlloc = asm["stackAlloc"];
  asm["stackAlloc"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_stackAlloc.apply(null, arguments);
  };

  var real_stackRestore = asm["stackRestore"];
  asm["stackRestore"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_stackRestore.apply(null, arguments);
  };

  var real___growWasmMemory = asm["__growWasmMemory"];
  asm["__growWasmMemory"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real___growWasmMemory.apply(null, arguments);
  };

  var real_dynCall_vi = asm["dynCall_vi"];
  asm["dynCall_vi"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_vi.apply(null, arguments);
  };

  var real_dynCall_ff = asm["dynCall_ff"];
  asm["dynCall_ff"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_ff.apply(null, arguments);
  };

  var real_dynCall_fff = asm["dynCall_fff"];
  asm["dynCall_fff"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_fff.apply(null, arguments);
  };

  var real_dynCall_viiiiiiiiiiii = asm["dynCall_viiiiiiiiiiii"];
  asm["dynCall_viiiiiiiiiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_viiiiiiiiiiii.apply(null, arguments);
  };

  var real_dynCall_viiiiiiiiii = asm["dynCall_viiiiiiiiii"];
  asm["dynCall_viiiiiiiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_viiiiiiiiii.apply(null, arguments);
  };

  var real_dynCall_viiii = asm["dynCall_viiii"];
  asm["dynCall_viiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_viiii.apply(null, arguments);
  };

  var real_dynCall_viiiiii = asm["dynCall_viiiiii"];
  asm["dynCall_viiiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_viiiiii.apply(null, arguments);
  };

  var real_dynCall_viiiiiiiii = asm["dynCall_viiiiiiiii"];
  asm["dynCall_viiiiiiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_viiiiiiiii.apply(null, arguments);
  };

  var real_dynCall_viiiiiiii = asm["dynCall_viiiiiiii"];
  asm["dynCall_viiiiiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_viiiiiiii.apply(null, arguments);
  };

  var real_dynCall_viiiii = asm["dynCall_viiiii"];
  asm["dynCall_viiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_viiiii.apply(null, arguments);
  };

  var real_dynCall_viiiiiiiiiii = asm["dynCall_viiiiiiiiiii"];
  asm["dynCall_viiiiiiiiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_viiiiiiiiiii.apply(null, arguments);
  };

  var real_dynCall_viiiiiii = asm["dynCall_viiiiiii"];
  asm["dynCall_viiiiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_viiiiiii.apply(null, arguments);
  };

  var real_dynCall_iiii = asm["dynCall_iiii"];
  asm["dynCall_iiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_iiii.apply(null, arguments);
  };

  var real_dynCall_iidiiii = asm["dynCall_iidiiii"];
  asm["dynCall_iidiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_iidiiii.apply(null, arguments);
  };

  var real_dynCall_jiji = asm["dynCall_jiji"];
  asm["dynCall_jiji"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return real_dynCall_jiji.apply(null, arguments);
  };

  Module["asm"] = asm;
  var ___wasm_call_ctors = Module["___wasm_call_ctors"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["__wasm_call_ctors"].apply(null, arguments)
  };

  var _init = Module["_init"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["init"].apply(null, arguments)
  };

  var _register_tensor = Module["_register_tensor"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["register_tensor"].apply(null, arguments)
  };

  var _dispose_data = Module["_dispose_data"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dispose_data"].apply(null, arguments)
  };

  var _dispose = Module["_dispose"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dispose"].apply(null, arguments)
  };

  var _Abs = Module["_Abs"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["Abs"].apply(null, arguments)
  };

  var _Add = Module["_Add"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["Add"].apply(null, arguments)
  };

  var _AddN = Module["_AddN"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["AddN"].apply(null, arguments)
  };

  var _AvgPool = Module["_AvgPool"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["AvgPool"].apply(null, arguments)
  };

  var _BatchMatMul = Module["_BatchMatMul"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["BatchMatMul"].apply(null, arguments)
  };

  var _ClipByValue = Module["_ClipByValue"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["ClipByValue"].apply(null, arguments)
  };

  var _Conv2D = Module["_Conv2D"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["Conv2D"].apply(null, arguments)
  };

  var _CropAndResize = Module["_CropAndResize"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["CropAndResize"].apply(null, arguments)
  };

  var _DepthwiseConv2dNative = Module["_DepthwiseConv2dNative"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["DepthwiseConv2dNative"].apply(null, arguments)
  };

  var _Div = Module["_Div"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["Div"].apply(null, arguments)
  };

  var _FusedBatchNorm = Module["_FusedBatchNorm"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["FusedBatchNorm"].apply(null, arguments)
  };

  var _FusedConv2D = Module["_FusedConv2D"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["FusedConv2D"].apply(null, arguments)
  };

  var _Max = Module["_Max"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["Max"].apply(null, arguments)
  };

  var _MaxPool = Module["_MaxPool"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["MaxPool"].apply(null, arguments)
  };

  var _Min = Module["_Min"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["Min"].apply(null, arguments)
  };

  var _Mul = Module["_Mul"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["Mul"].apply(null, arguments)
  };

  var _PadV2 = Module["_PadV2"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["PadV2"].apply(null, arguments)
  };

  var _Prelu = Module["_Prelu"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["Prelu"].apply(null, arguments)
  };

  var _Sigmoid = Module["_Sigmoid"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["Sigmoid"].apply(null, arguments)
  };

  var _Square = Module["_Square"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["Square"].apply(null, arguments)
  };

  var _Sub = Module["_Sub"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["Sub"].apply(null, arguments)
  };

  var _Transpose = Module["_Transpose"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["Transpose"].apply(null, arguments)
  };

  var _malloc = Module["_malloc"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["malloc"].apply(null, arguments)
  };

  var _free = Module["_free"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["free"].apply(null, arguments)
  };

  var _setThrew = Module["_setThrew"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["setThrew"].apply(null, arguments)
  };

  var ___cxa_demangle = Module["___cxa_demangle"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["__cxa_demangle"].apply(null, arguments)
  };

  var dynCall_i = Module["dynCall_i"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_i"].apply(null, arguments)
  };

  var dynCall_ii = Module["dynCall_ii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_ii"].apply(null, arguments)
  };

  var dynCall_iii = Module["dynCall_iii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_iii"].apply(null, arguments)
  };

  var dynCall_v = Module["dynCall_v"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_v"].apply(null, arguments)
  };

  var dynCall_vii = Module["dynCall_vii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_vii"].apply(null, arguments)
  };

  var dynCall_viii = Module["dynCall_viii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_viii"].apply(null, arguments)
  };

  var stackSave = Module["stackSave"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["stackSave"].apply(null, arguments)
  };

  var stackAlloc = Module["stackAlloc"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["stackAlloc"].apply(null, arguments)
  };

  var stackRestore = Module["stackRestore"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["stackRestore"].apply(null, arguments)
  };

  var __growWasmMemory = Module["__growWasmMemory"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["__growWasmMemory"].apply(null, arguments)
  };

  var dynCall_vi = Module["dynCall_vi"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_vi"].apply(null, arguments)
  };

  var dynCall_ff = Module["dynCall_ff"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_ff"].apply(null, arguments)
  };

  var dynCall_fff = Module["dynCall_fff"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_fff"].apply(null, arguments)
  };

  var dynCall_viiiiiiiiiiii = Module["dynCall_viiiiiiiiiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_viiiiiiiiiiii"].apply(null, arguments)
  };

  var dynCall_viiiiiiiiii = Module["dynCall_viiiiiiiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_viiiiiiiiii"].apply(null, arguments)
  };

  var dynCall_viiii = Module["dynCall_viiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_viiii"].apply(null, arguments)
  };

  var dynCall_viiiiii = Module["dynCall_viiiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_viiiiii"].apply(null, arguments)
  };

  var dynCall_viiiiiiiii = Module["dynCall_viiiiiiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_viiiiiiiii"].apply(null, arguments)
  };

  var dynCall_viiiiiiii = Module["dynCall_viiiiiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_viiiiiiii"].apply(null, arguments)
  };

  var dynCall_viiiii = Module["dynCall_viiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_viiiii"].apply(null, arguments)
  };

  var dynCall_viiiiiiiiiii = Module["dynCall_viiiiiiiiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_viiiiiiiiiii"].apply(null, arguments)
  };

  var dynCall_viiiiiii = Module["dynCall_viiiiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_viiiiiii"].apply(null, arguments)
  };

  var dynCall_iiii = Module["dynCall_iiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_iiii"].apply(null, arguments)
  };

  var dynCall_iidiiii = Module["dynCall_iidiiii"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_iidiiii"].apply(null, arguments)
  };

  var dynCall_jiji = Module["dynCall_jiji"] = function() {
    assert(runtimeInitialized, 'you need to wait for the runtime to be ready (e.g. wait for main() to be called)');
    assert(!runtimeExited, 'the runtime was exited (use NO_EXIT_RUNTIME to keep it alive after main() exits)');
    return Module["asm"]["dynCall_jiji"].apply(null, arguments)
  };


  function invoke_iii(index,a1,a2) {
    var sp = stackSave();
    try {
      return dynCall_iii(index,a1,a2);
    } catch(e) {
      stackRestore(sp);
      if (e !== e+0 && e !== 'longjmp') throw e;
      _setThrew(1, 0);
    }
  }

  function invoke_v(index) {
    var sp = stackSave();
    try {
      dynCall_v(index);
    } catch(e) {
      stackRestore(sp);
      if (e !== e+0 && e !== 'longjmp') throw e;
      _setThrew(1, 0);
    }
  }

  function invoke_ii(index,a1) {
    var sp = stackSave();
    try {
      return dynCall_ii(index,a1);
    } catch(e) {
      stackRestore(sp);
      if (e !== e+0 && e !== 'longjmp') throw e;
      _setThrew(1, 0);
    }
  }

  function invoke_vii(index,a1,a2) {
    var sp = stackSave();
    try {
      dynCall_vii(index,a1,a2);
    } catch(e) {
      stackRestore(sp);
      if (e !== e+0 && e !== 'longjmp') throw e;
      _setThrew(1, 0);
    }
  }

  function invoke_viii(index,a1,a2,a3) {
    var sp = stackSave();
    try {
      dynCall_viii(index,a1,a2,a3);
    } catch(e) {
      stackRestore(sp);
      if (e !== e+0 && e !== 'longjmp') throw e;
      _setThrew(1, 0);
    }
  }

  function invoke_i(index) {
    var sp = stackSave();
    try {
      return dynCall_i(index);
    } catch(e) {
      stackRestore(sp);
      if (e !== e+0 && e !== 'longjmp') throw e;
      _setThrew(1, 0);
    }
  }



  // === Auto-generated postamble setup entry stuff ===

  Module['asm'] = asm;

  if (!Object.getOwnPropertyDescriptor(Module, "intArrayFromString")) Module["intArrayFromString"] = function() { abort("'intArrayFromString' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "intArrayToString")) Module["intArrayToString"] = function() { abort("'intArrayToString' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "ccall")) Module["ccall"] = function() { abort("'ccall' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  Module["cwrap"] = cwrap;
  if (!Object.getOwnPropertyDescriptor(Module, "setValue")) Module["setValue"] = function() { abort("'setValue' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "getValue")) Module["getValue"] = function() { abort("'getValue' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "allocate")) Module["allocate"] = function() { abort("'allocate' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "getMemory")) Module["getMemory"] = function() { abort("'getMemory' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"); };
  if (!Object.getOwnPropertyDescriptor(Module, "AsciiToString")) Module["AsciiToString"] = function() { abort("'AsciiToString' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "stringToAscii")) Module["stringToAscii"] = function() { abort("'stringToAscii' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "UTF8ArrayToString")) Module["UTF8ArrayToString"] = function() { abort("'UTF8ArrayToString' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "UTF8ToString")) Module["UTF8ToString"] = function() { abort("'UTF8ToString' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "stringToUTF8Array")) Module["stringToUTF8Array"] = function() { abort("'stringToUTF8Array' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "stringToUTF8")) Module["stringToUTF8"] = function() { abort("'stringToUTF8' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "lengthBytesUTF8")) Module["lengthBytesUTF8"] = function() { abort("'lengthBytesUTF8' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "UTF16ToString")) Module["UTF16ToString"] = function() { abort("'UTF16ToString' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "stringToUTF16")) Module["stringToUTF16"] = function() { abort("'stringToUTF16' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "lengthBytesUTF16")) Module["lengthBytesUTF16"] = function() { abort("'lengthBytesUTF16' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "UTF32ToString")) Module["UTF32ToString"] = function() { abort("'UTF32ToString' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "stringToUTF32")) Module["stringToUTF32"] = function() { abort("'stringToUTF32' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "lengthBytesUTF32")) Module["lengthBytesUTF32"] = function() { abort("'lengthBytesUTF32' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "allocateUTF8")) Module["allocateUTF8"] = function() { abort("'allocateUTF8' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "stackTrace")) Module["stackTrace"] = function() { abort("'stackTrace' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "addOnPreRun")) Module["addOnPreRun"] = function() { abort("'addOnPreRun' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "addOnInit")) Module["addOnInit"] = function() { abort("'addOnInit' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "addOnPreMain")) Module["addOnPreMain"] = function() { abort("'addOnPreMain' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "addOnExit")) Module["addOnExit"] = function() { abort("'addOnExit' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "addOnPostRun")) Module["addOnPostRun"] = function() { abort("'addOnPostRun' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "writeStringToMemory")) Module["writeStringToMemory"] = function() { abort("'writeStringToMemory' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "writeArrayToMemory")) Module["writeArrayToMemory"] = function() { abort("'writeArrayToMemory' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "writeAsciiToMemory")) Module["writeAsciiToMemory"] = function() { abort("'writeAsciiToMemory' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "addRunDependency")) Module["addRunDependency"] = function() { abort("'addRunDependency' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"); };
  if (!Object.getOwnPropertyDescriptor(Module, "removeRunDependency")) Module["removeRunDependency"] = function() { abort("'removeRunDependency' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"); };
  if (!Object.getOwnPropertyDescriptor(Module, "ENV")) Module["ENV"] = function() { abort("'ENV' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "FS")) Module["FS"] = function() { abort("'FS' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "FS_createFolder")) Module["FS_createFolder"] = function() { abort("'FS_createFolder' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"); };
  if (!Object.getOwnPropertyDescriptor(Module, "FS_createPath")) Module["FS_createPath"] = function() { abort("'FS_createPath' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"); };
  if (!Object.getOwnPropertyDescriptor(Module, "FS_createDataFile")) Module["FS_createDataFile"] = function() { abort("'FS_createDataFile' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"); };
  if (!Object.getOwnPropertyDescriptor(Module, "FS_createPreloadedFile")) Module["FS_createPreloadedFile"] = function() { abort("'FS_createPreloadedFile' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"); };
  if (!Object.getOwnPropertyDescriptor(Module, "FS_createLazyFile")) Module["FS_createLazyFile"] = function() { abort("'FS_createLazyFile' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"); };
  if (!Object.getOwnPropertyDescriptor(Module, "FS_createLink")) Module["FS_createLink"] = function() { abort("'FS_createLink' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"); };
  if (!Object.getOwnPropertyDescriptor(Module, "FS_createDevice")) Module["FS_createDevice"] = function() { abort("'FS_createDevice' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"); };
  if (!Object.getOwnPropertyDescriptor(Module, "FS_unlink")) Module["FS_unlink"] = function() { abort("'FS_unlink' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"); };
  if (!Object.getOwnPropertyDescriptor(Module, "GL")) Module["GL"] = function() { abort("'GL' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "dynamicAlloc")) Module["dynamicAlloc"] = function() { abort("'dynamicAlloc' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "loadDynamicLibrary")) Module["loadDynamicLibrary"] = function() { abort("'loadDynamicLibrary' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "loadWebAssemblyModule")) Module["loadWebAssemblyModule"] = function() { abort("'loadWebAssemblyModule' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "getLEB")) Module["getLEB"] = function() { abort("'getLEB' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "getFunctionTables")) Module["getFunctionTables"] = function() { abort("'getFunctionTables' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "alignFunctionTables")) Module["alignFunctionTables"] = function() { abort("'alignFunctionTables' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "registerFunctions")) Module["registerFunctions"] = function() { abort("'registerFunctions' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "addFunction")) Module["addFunction"] = function() { abort("'addFunction' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "removeFunction")) Module["removeFunction"] = function() { abort("'removeFunction' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "getFuncWrapper")) Module["getFuncWrapper"] = function() { abort("'getFuncWrapper' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "prettyPrint")) Module["prettyPrint"] = function() { abort("'prettyPrint' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "makeBigInt")) Module["makeBigInt"] = function() { abort("'makeBigInt' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "dynCall")) Module["dynCall"] = function() { abort("'dynCall' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "getCompilerSetting")) Module["getCompilerSetting"] = function() { abort("'getCompilerSetting' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "stackSave")) Module["stackSave"] = function() { abort("'stackSave' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "stackRestore")) Module["stackRestore"] = function() { abort("'stackRestore' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "stackAlloc")) Module["stackAlloc"] = function() { abort("'stackAlloc' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "establishStackSpace")) Module["establishStackSpace"] = function() { abort("'establishStackSpace' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "print")) Module["print"] = function() { abort("'print' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "printErr")) Module["printErr"] = function() { abort("'printErr' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "getTempRet0")) Module["getTempRet0"] = function() { abort("'getTempRet0' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "setTempRet0")) Module["setTempRet0"] = function() { abort("'setTempRet0' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "callMain")) Module["callMain"] = function() { abort("'callMain' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "abort")) Module["abort"] = function() { abort("'abort' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "Pointer_stringify")) Module["Pointer_stringify"] = function() { abort("'Pointer_stringify' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  if (!Object.getOwnPropertyDescriptor(Module, "warnOnce")) Module["warnOnce"] = function() { abort("'warnOnce' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); };
  Module["writeStackCookie"] = writeStackCookie;
  Module["checkStackCookie"] = checkStackCookie;
  Module["abortStackOverflow"] = abortStackOverflow;if (!Object.getOwnPropertyDescriptor(Module, "ALLOC_NORMAL")) Object.defineProperty(Module, "ALLOC_NORMAL", { configurable: true, get: function() { abort("'ALLOC_NORMAL' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); } });
  if (!Object.getOwnPropertyDescriptor(Module, "ALLOC_STACK")) Object.defineProperty(Module, "ALLOC_STACK", { configurable: true, get: function() { abort("'ALLOC_STACK' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); } });
  if (!Object.getOwnPropertyDescriptor(Module, "ALLOC_DYNAMIC")) Object.defineProperty(Module, "ALLOC_DYNAMIC", { configurable: true, get: function() { abort("'ALLOC_DYNAMIC' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); } });
  if (!Object.getOwnPropertyDescriptor(Module, "ALLOC_NONE")) Object.defineProperty(Module, "ALLOC_NONE", { configurable: true, get: function() { abort("'ALLOC_NONE' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ)"); } });
  if (!Object.getOwnPropertyDescriptor(Module, "calledRun")) Object.defineProperty(Module, "calledRun", { configurable: true, get: function() { abort("'calledRun' was not exported. add it to EXTRA_EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"); } });



  var calledRun;

  // Modularize mode returns a function, which can be called to
  // create instances. The instances provide a then() method,
  // must like a Promise, that receives a callback. The callback
  // is called when the module is ready to run, with the module
  // as a parameter. (Like a Promise, it also returns the module
  // so you can use the output of .then(..)).
  Module['then'] = function(func) {
    // We may already be ready to run code at this time. if
    // so, just queue a call to the callback.
    if (calledRun) {
      func(Module);
    } else {
      // we are not ready to call then() yet. we must call it
      // at the same time we would call onRuntimeInitialized.
      var old = Module['onRuntimeInitialized'];
      Module['onRuntimeInitialized'] = function() {
        if (old) old();
        func(Module);
      };
    }
    return Module;
  };


  dependenciesFulfilled = function runCaller() {
    // If run has never been called, and we should call run (INVOKE_RUN is true, and Module.noInitialRun is not false)
    if (!calledRun) run();
    if (!calledRun) dependenciesFulfilled = runCaller; // try this again later, after new deps are fulfilled
  };





  /** @type {function(Array=)} */
  function run(args) {

    if (runDependencies > 0) {
      return;
    }

    writeStackCookie();

    preRun();

    if (runDependencies > 0) return; // a preRun added a dependency, run will be called later

    function doRun() {
      // run may have just been called through dependencies being fulfilled just in this very frame,
      // or while the async setStatus time below was happening
      if (calledRun) return;
      calledRun = true;

      if (ABORT) return;

      initRuntime();

      preMain();

      if (Module['onRuntimeInitialized']) Module['onRuntimeInitialized']();

      assert(!Module['_main'], 'compiled without a main, but one is present. if you added it from JS, use Module["onRuntimeInitialized"]');

      postRun();
    }

    if (Module['setStatus']) {
      Module['setStatus']('Running...');
      setTimeout(function() {
        setTimeout(function() {
          Module['setStatus']('');
        }, 1);
        doRun();
      }, 1);
    } else
    {
      doRun();
    }
    checkStackCookie();
  }
  Module['run'] = run;

  if (Module['preInit']) {
    if (typeof Module['preInit'] == 'function') Module['preInit'] = [Module['preInit']];
    while (Module['preInit'].length > 0) {
      Module['preInit'].pop()();
    }
  }


    noExitRuntime = true;

  run();





  // {{MODULE_ADDITIONS}}





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
      BackendWasm.prototype.makeOutput = function (shape, dtype) {
          var dataId = this.write(null /* values */, shape, dtype);
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
