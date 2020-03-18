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

// We explicitly import the modular kernels so they get registered in the
// global registry when we compile the library. A modular build would replace
// the contents of this file and import only the kernels that are needed.
import './_FusedMatMul';
import './Abs';
import './Add';
import './AddN';
import './ArgMax';
import './AvgPool';
import './BatchMatMul';
import './Cast';
import './ClipByValue';
import './Concat';
import './Conv2D';
import './Cos';
import './CropAndResize';
import './DepthwiseConv2dNative';
import './Div';
import './Exp';
import './FloorDiv';
import './FusedBatchNorm';
import './FusedConv2D';
import './FusedDepthwiseConv2D';
import './Gather';
import './GatherNd';
import './Greater';
import './GreaterEqual';
import './Less';
import './LessEqual';
import './Log';
import './LogicalAnd';
import './Max';
import './Maximum';
import './MaxPool';
import './Min';
import './Minimum';
import './Mul';
import './Neg';
import './NonMaxSuppressionV3';
import './NonMaxSuppressionV5';
import './NotEqual';
import './OnesLike';
import './PadV2';
import './Pow';
import './Prelu';
import './Relu';
import './Relu6';
import './Reshape';
import './ResizeBilinear';
import './Rsqrt';
import './ScatterNd';
import './Sigmoid';
import './Sin';
import './Slice';
import './Softmax';
import './Square';
import './Sub';
import './Sum';
import './Tanh';
import './Tile';
import './Transpose';
import './Unpack';
import './ZerosLike';
