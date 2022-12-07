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

import {KernelConfig, scalar, UnsortedSegmentSum, UnsortedSegmentSumAttrs, UnsortedSegmentSumInputs} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const unsortedSegmentSumConfig: KernelConfig = {
  kernelName: UnsortedSegmentSum,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x, segmentIds} = args.inputs as UnsortedSegmentSumInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {numSegments} = args.attrs as unknown as UnsortedSegmentSumAttrs;

    const opAttrs = [
      createTensorsTypeOpAttr('T', x.dtype),
      createTensorsTypeOpAttr('Tindices', 'int32'),
      createTensorsTypeOpAttr('Tnumsegments', 'int32')
    ];
    const numSegmentsT = scalar(numSegments, 'int32');
    const res = backend.executeSingleOutput(
        UnsortedSegmentSum, opAttrs, [x, segmentIds, numSegmentsT]);
    numSegmentsT.dispose();
    return res;
  }
};
