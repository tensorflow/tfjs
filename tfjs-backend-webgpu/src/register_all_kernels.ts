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
import {KernelConfig, registerKernel} from '@tensorflow/tfjs-core';

import { registerKernelsBinary } from './kernels/Binary';
import { registerKernelsUnary } from './kernels/Unary';

import {_fusedMatMulConfig} from './kernels/_FusedMatMul';
import {addNConfig} from './kernels/AddN';
import {argMaxConfig} from './kernels/ArgMax';
import {argMinConfig} from './kernels/ArgMin';
import {avgPoolConfig} from './kernels/AvgPool';
import {batchMatMulConfig} from './kernels/BatchMatMul';
import {batchToSpaceNDConfig} from './kernels/BatchToSpaceND';
import {castConfig} from './kernels/Cast';
import {clipByValueConfig} from './kernels/ClipByValue';
import {concatConfig} from './kernels/Concat';
import {conv2DConfig} from './kernels/Conv2D';
import {conv2DBackpropInputConfig} from './kernels/Conv2DBackpropInput';
import {cropAndResizeConfig} from './kernels/CropAndResize';
import {depthToSpaceConfig} from './kernels/DepthToSpace';
import {depthwiseConv2dNativeConfig} from './kernels/DepthwiseConv2dNative';
import {einsumConfig} from './kernels/Einsum';
import {expandDimsConfig} from './kernels/ExpandDims';
import {fillConfig} from './kernels/Fill';
import {flipLeftRightConfig} from './kernels/FlipLeftRight';
import {fromPixelsConfig} from './kernels/FromPixels';
import {fusedBatchNormConfig} from './kernels/FusedBatchNorm';
import {fusedConv2DConfig} from './kernels/FusedConv2D';
import {fusedDepthwiseConv2DConfig} from './kernels/FusedDepthwiseConv2D';
import {gatherNdConfig} from './kernels/GatherNd';
import {gatherV2Config} from './kernels/GatherV2';
import {identityConfig} from './kernels/Identity';
import {imagConfig} from './kernels/Imag';
import {maxConfig} from './kernels/Max';
import {maxPoolConfig} from './kernels/MaxPool';
import {meanConfig} from './kernels/Mean';
import {minConfig} from './kernels/Min';
import {mirrorPadConfig} from './kernels/MirrorPad';
import {nonMaxSuppressionV3Config} from './kernels/NonMaxSuppressionV3';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {onesLikeConfig} from './kernels/OnesLike';
import {packConfig} from './kernels/Pack';
import {padV2Config} from './kernels/PadV2';
import {prodConfig} from './kernels/Prod';
import {rangeConfig} from './kernels/Range';
import {realConfig} from './kernels/Real';
import {reshapeConfig} from './kernels/Reshape';
import {resizeBilinearConfig} from './kernels/ResizeBilinear';
import {resizeNearestNeighborConfig} from './kernels/ResizeNearestNeighbor';
import {rotateWithOffsetConfig} from './kernels/RotateWithOffset';
import {scatterNdConfig} from './kernels/ScatterNd';
import {selectConfig} from './kernels/Select';
import {sliceConfig} from './kernels/Slice';
import {softmaxConfig} from './kernels/Softmax';
import {spaceToBatchNDConfig} from './kernels/SpaceToBatchND';
import {sparseToDenseConfig} from './kernels/SparseToDense';
import {splitVConfig} from './kernels/SplitV';
import {stridedSliceConfig} from './kernels/StridedSlice';
import {stringNGramsConfig} from './kernels/StringNGrams';
import {sumConfig} from './kernels/Sum';
import {tileConfig} from './kernels/Tile';
import { topKConfig } from './kernels/TopK';
import {transformConfig} from './kernels/Transform';
import {transposeConfig} from './kernels/Transpose';
import {unpackConfig} from './kernels/Unpack';
import {zerosLikeConfig} from './kernels/ZerosLike';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  _fusedMatMulConfig,
  addNConfig,
  argMaxConfig,
  argMinConfig,
  avgPoolConfig,
  batchMatMulConfig,
  batchToSpaceNDConfig,
  castConfig,
  clipByValueConfig,
  concatConfig,
  conv2DConfig,
  conv2DBackpropInputConfig,
  cropAndResizeConfig,
  depthToSpaceConfig,
  depthwiseConv2dNativeConfig,
  einsumConfig,
  expandDimsConfig,
  fillConfig,
  flipLeftRightConfig,
  fromPixelsConfig,
  fusedBatchNormConfig,
  fusedConv2DConfig,
  fusedDepthwiseConv2DConfig,
  gatherNdConfig,
  gatherV2Config,
  identityConfig,
  imagConfig,
  maxConfig,
  maxPoolConfig,
  meanConfig,
  minConfig,
  mirrorPadConfig,
  nonMaxSuppressionV3Config,
  nonMaxSuppressionV5Config,
  onesLikeConfig,
  packConfig,
  padV2Config,
  prodConfig,
  rangeConfig,
  realConfig,
  reshapeConfig,
  resizeBilinearConfig,
  resizeNearestNeighborConfig,
  rotateWithOffsetConfig,
  scatterNdConfig,
  selectConfig,
  sliceConfig,
  stridedSliceConfig,
  stringNGramsConfig,
  softmaxConfig,
  spaceToBatchNDConfig,
  splitVConfig,
  sparseToDenseConfig,
  sumConfig,
  tileConfig,
  topKConfig,
  transformConfig,
  transposeConfig,
  unpackConfig,
  zerosLikeConfig
];

registerKernelsBinary();
registerKernelsUnary();

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
