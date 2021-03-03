/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap,
     context: ExecutionContext): Tensor[] => {
      switch (node.op) {
        case 'ResizeBilinear': {
          const images =
              getParamValue('images', node, tensorMap, context) as Tensor;
          const size =
              getParamValue('size', node, tensorMap, context) as number[];
          const alignCorners =
              getParamValue('alignCorners', node, tensorMap, context) as
              boolean;
          const halfPixelCenters =
              getParamValue('halfPixelCenters', node, tensorMap, context) as
              boolean;
          return [tfOps.image.resizeBilinear(
              images as Tensor3D | Tensor4D, [size[0], size[1]], alignCorners,
              halfPixelCenters)];
        }
        case 'ResizeNearestNeighbor': {
          const images =
              getParamValue('images', node, tensorMap, context) as Tensor;
          const size =
              getParamValue('size', node, tensorMap, context) as number[];
          const alignCorners =
              getParamValue('alignCorners', node, tensorMap, context) as
              boolean;
          const halfPixelCenters =
              getParamValue('halfPixelCenters', node, tensorMap, context) as
              boolean;
          return [tfOps.image.resizeNearestNeighbor(
              images as Tensor3D | Tensor4D, [size[0], size[1]], alignCorners,
              halfPixelCenters)];
        }
        case 'CropAndResize': {
          const image =
              getParamValue('image', node, tensorMap, context) as Tensor;
          const boxes =
              getParamValue('boxes', node, tensorMap, context) as Tensor;
          const boxInd =
              getParamValue('boxInd', node, tensorMap, context) as Tensor;
          const cropSize =
              getParamValue('cropSize', node, tensorMap, context) as number[];
          const method =
              getParamValue('method', node, tensorMap, context) as string;
          const extrapolationValue =
              getParamValue('extrapolationValue', node, tensorMap, context) as
              number;
          return [tfOps.image.cropAndResize(
              image as Tensor4D, boxes as Tensor2D, boxInd as Tensor1D,
              cropSize as [number, number], method as 'bilinear' | 'nearest',
              extrapolationValue)];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'image';
