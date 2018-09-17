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

export const json = [
  {
    'tfOpName': 'ConcatV2',
    'dlOpName': 'concat',
    'category': 'slice_join',
    'params': [
      {
        'tfInputIndex': 0,
        'tfInputParamLength': 1,
        'dlParamName': 'tensors',
        'type': 'tensors'
      },
      {'tfInputIndex': -1, 'dlParamName': 'axis', 'type': 'number'}
    ]
  },
  {
    'tfOpName': 'Concat',
    'dlOpName': 'concat',
    'category': 'slice_join',
    'params': [
      {
        'tfInputIndex': 1,
        'tfInputParamLength': 1,
        'dlParamName': 'tensors',
        'type': 'tensors'
      },
      {'tfInputIndex': 0, 'dlParamName': 'axis', 'type': 'number'}
    ]
  },
  {
    'tfOpName': 'GatherV2',
    'dlOpName': 'gather',
    'category': 'slice_join',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'indices', 'type': 'tensor'}, {
        'tfParamName': 'axis',
        'dlParamName': 'axis',
        'type': 'number',
        'defaultValue': 0
      }
    ]
  },
  {
    'tfOpName': 'Gather',
    'dlOpName': 'gather',
    'category': 'slice_join',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'indices', 'type': 'tensor'}, {
        'tfParamName': 'axis',
        'dlParamName': 'axis',
        'type': 'number',
        'defaultValue': 0
      },
      {
        'tfParamName': 'validate_indices',
        'dlParamName': 'validateIndices',
        'type': 'bool',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'Reverse',
    'dlOpName': 'reverse',
    'category': 'slice_join',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'axis', 'type': 'number'}
    ]
  },
  {
    'tfOpName': 'ReverseV2',
    'dlOpName': 'reverse',
    'category': 'slice_join',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'axis', 'type': 'number'}
    ]
  },
  {
    'tfOpName': 'Slice',
    'dlOpName': 'slice',
    'category': 'slice_join',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'begin', 'type': 'number[]'},
      {'tfInputIndex': 2, 'dlParamName': 'size', 'type': 'number[]'}
    ]
  },
  {
    'tfOpName': 'StridedSlice',
    'dlOpName': 'stridedSlice',
    'category': 'slice_join',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'begin', 'type': 'number[]'},
      {'tfInputIndex': 2, 'dlParamName': 'end', 'type': 'number[]'},
      {'tfInputIndex': 3, 'dlParamName': 'strides', 'type': 'number[]'}, {
        'tfParamName': 'begin_mask',
        'dlParamName': 'beginMask',
        'type': 'number',
        'defaultValue': 0
      },
      {
        'tfParamName': 'end_mask',
        'dlParamName': 'endMask',
        'type': 'number',
        'defaultValue': 0
      },
      {
        'tfParamName': 'new_axis_mask',
        'dlParamName': 'newAxisMask',
        'type': 'number',
        'defaultValue': 0
      },
      {
        'tfParamName': 'ellipsis_mask',
        'dlParamName': 'ellipsisMask',
        'type': 'number',
        'defaultValue': 0
      },
      {
        'tfParamName': 'shrink_axis_mask',
        'dlParamName': 'shrinkAxisMask',
        'type': 'number',
        'defaultValue': 0
      }
    ]
  },
  {
    'tfOpName': 'Pack',
    'dlOpName': 'stack',
    'category': 'slice_join',
    'params': [
      {
        'tfInputIndex': 0,
        'tfInputParamLength': 0,
        'dlParamName': 'tensors',
        'type': 'tensors'
      },
      {
        'tfParamName': 'axis',
        'dlParamName': 'axis',
        'type': 'number',
        'defaultValue': 0
      }
    ]
  },
  {
    'tfOpName': 'Unpack',
    'dlOpName': 'unstack',
    'category': 'slice_join',
    'params': [
      {
        'tfInputIndex': 0,
        'tfInputParamLength': 0,
        'dlParamName': 'tensor',
        'type': 'tensor'
      },
      {
        'tfParamName': 'axis',
        'dlParamName': 'axis',
        'type': 'number',
        'defaultValue': 0
      },
      {
        'tfParamName': 'num',
        'dlParamName': 'num',
        'type': 'number',
        'defaultValue': 0,
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'Tile',
    'dlOpName': 'tile',
    'category': 'slice_join',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'reps', 'type': 'number[]'}
    ]
  },
  {
    'tfOpName': 'Split',
    'dlOpName': 'split',
    'category': 'slice_join',
    'params': [
      {
        'tfInputIndex': 0,
        'dlParamName': 'axis',
        'type': 'number',
        'defaultValue': 0
      },
      {'tfInputIndex': 1, 'dlParamName': 'x', 'type': 'tensor'}, {
        'tfParamName': 'num_split',
        'dlParamName': 'numOrSizeSplits',
        'type': 'number',
        'defaultValue': 1
      }
    ]
  },
  {
    'tfOpName': 'SplitV',
    'dlOpName': 'split',
    'category': 'slice_join',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'numOrSizeSplits', 'type': 'number[]'},
      {
        'tfInputIndex': 2,
        'dlParamName': 'axis',
        'type': 'number',
        'defaultValue': 0
      }
    ]
  }
];
