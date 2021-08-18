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

import {backend_util, KernelConfig, reshape, slice, StridedSlice, StridedSliceAttrs, StridedSliceInputs, Tensor, tensor1d, tidy} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const stridedSliceConfig: KernelConfig = {
  kernelName: StridedSlice,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    let {x} = args.inputs as StridedSliceInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask} =
        args.attrs as {} as StridedSliceAttrs;

    let {begin, end, strides} = args.attrs as {} as StridedSliceAttrs;
    // make a copy because it may be modified in-place further down.
    begin = begin.slice();
    end = end.slice();
    if (strides == null) {
      strides = new Array(begin.length);
    }

    const ellipsisAxes = backend_util.slice_util.maskToAxes(ellipsisMask);
    if (ellipsisAxes.length > 1) {
      throw new Error('Multiple ellipses in slice is not allowed.');
    }

    if (ellipsisMask !== 0 && newAxisMask !== 0) {
      throw new Error(
          'Using both ellipsisMask and newAxisMask is not yet supported.');
    }

    if (ellipsisMask !== 0 && shrinkAxisMask !== 0) {
      throw new Error(
          'Using both ellipsisMask and shrinkAxisMask is not yet supported.');
    }

    return tidy(() => {
      const numInterpolatedAxes = (x as Tensor).rank - begin.length;

      // Expand the dims of x based on the newAxisMask.
      const expandAxes = backend_util.slice_util.maskToAxes(newAxisMask);
      const newShape = x.shape.slice();
      expandAxes.forEach(axis => {
        begin[axis] = 0;
        end[axis] = 1;
        newShape.splice(axis, 0, 1);
      });
      x = reshape(x as Tensor, newShape);

      const {
        begin: normalizedBegin,
        end: normalizedEnd,
        strides: normalizedStrides
      } =
          backend_util.slice_util.getNormalizedAxes(
              x.shape, ellipsisAxes, numInterpolatedAxes, begin, end, strides,
              beginMask, endMask, ellipsisMask);
      begin = normalizedBegin;
      end = normalizedEnd;
      strides = normalizedStrides;

      const shrinkAxes = backend_util.slice_util.maskToAxes(shrinkAxisMask);
      // Adjust the ends based on the shrink mask.
      shrinkAxes.forEach(axis => {
        end[axis] = begin[axis] + 1;
        strides[axis] = 1;
      });

      // Figure out the output shape.
      const size = backend_util.slice_util.computeOutShape(begin, end, strides);
      // Remove the axes based on shrinkMask.
      const outShape =
          size.filter((_, axis) => shrinkAxes.indexOf(axis) === -1);

      const nonStrided = strides.every(v => v === 1);
      if (nonStrided) {
        return reshape(slice(x as Tensor, begin, size), outShape);
      }

      const beginTensor = tensor1d(begin, 'int32');
      for (let axis = 0; axis < end.length; axis++) {
        // Unlike Numpy, when the strides are negative, TF C uses -n-1 instead
        // of -1 as the "end" in order to include the first element.
        if (strides[axis] < 0 && end[axis] === -1) {
          end[axis] -= x.shape[axis];
        }
      }
      const endTensor = tensor1d(end, 'int32');
      const stridesTensor = tensor1d(strides, 'int32');

      // Because tfjs allows begin, end and strides to be of different lengths
      // It has custom code (see above to handle the masking), so we do not
      // pass the mask info to the tensorflow backend.
      const opAttrs = [
        createTensorsTypeOpAttr('T', x.dtype),
        createTensorsTypeOpAttr('Index', 'int32'),
        {name: 'begin_mask', type: backend.binding.TF_ATTR_INT, value: 0},
        {name: 'end_mask', type: backend.binding.TF_ATTR_INT, value: 0},
        {name: 'ellipsis_mask', type: backend.binding.TF_ATTR_INT, value: 0},
        {name: 'new_axis_mask', type: backend.binding.TF_ATTR_INT, value: 0},
        {name: 'shrink_axis_mask', type: backend.binding.TF_ATTR_INT, value: 0}
      ];
      const res = backend.executeSingleOutput(
          StridedSlice, opAttrs, [x, beginTensor, endTensor, stridesTensor]);
      return reshape(res, outShape);
    });
  }
};
