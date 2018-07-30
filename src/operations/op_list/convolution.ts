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
    'tfOpName': 'AvgPool',
    'dlOpName': 'avgPool',
    'category': 'convolution',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfParamName': 'strides', 'dlParamName': 'strides', 'type': 'number[]'},
      {'tfParamName': 'padding', 'dlParamName': 'pad', 'type': 'string'}, {
        'tfParamName': 'data_format',
        'dlParamName': 'dataFormat',
        'type': 'string',
        'notSupported': true
      },
      {'tfParamName': 'ksize', 'dlParamName': 'kernelSize', 'type': 'number[]'},
      {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'MaxPool',
    'dlOpName': 'maxPool',
    'category': 'convolution',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfParamName': 'strides', 'dlParamName': 'strides', 'type': 'number[]'},
      {'tfParamName': 'padding', 'dlParamName': 'pad', 'type': 'string'}, {
        'tfParamName': 'data_format',
        'dlParamName': 'dataFormat',
        'type': 'string',
        'notSupported': true
      },
      {'tfParamName': 'ksize', 'dlParamName': 'kernelSize', 'type': 'number[]'},
      {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'Conv1D',
    'dlOpName': 'conv1d',
    'category': 'convolution',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'filter', 'type': 'tensor'},
      {'tfParamName': 'stride', 'dlParamName': 'stride', 'type': 'number'},
      {'tfParamName': 'padding', 'dlParamName': 'pad', 'type': 'string'}, {
        'tfParamName': 'data_format',
        'dlParamName': 'dataFormat',
        'type': 'string',
        'defaultValue': 'NWC'
      },
      {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      },
      {
        'tfParamName': 'dilation',
        'dlParamName': 'dilation',
        'type': 'number',
        'defaultValue': 1
      }
    ]
  },
  {
    'tfOpName': 'Conv2D',
    'dlOpName': 'conv2d',
    'category': 'convolution',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'filter', 'type': 'tensor'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      },
      {'tfParamName': 'strides', 'dlParamName': 'strides', 'type': 'number[]'},
      {'tfParamName': 'padding', 'dlParamName': 'pad', 'type': 'string'}, {
        'tfParamName': 'useCudnnOnGpu',
        'dlParamName': 'useCudnnOnGpu',
        'type': 'bool'
      },
      {
        'tfParamName': 'data_format',
        'dlParamName': 'dataFormat',
        'type': 'string',
        'defaultValue': 'NHWC'
      },
      {
        'tfParamName': 'dilations',
        'dlParamName': 'dilations',
        'type': 'number[]'
      }
    ]
  },
  {
    'tfOpName': 'Conv2DBackpropInput',
    'dlOpName': 'conv2dTranspose',
    'category': 'convolution',
    'params': [
      {'tfInputIndex': 2, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'filter', 'type': 'tensor'},
      {'tfInputIndex': 0, 'dlParamName': 'outputShape', 'type': 'number[]'},
      {'tfParamName': 'strides', 'dlParamName': 'strides', 'type': 'number[]'},
      {'tfParamName': 'padding', 'dlParamName': 'pad', 'type': 'string'}, {
        'tfParamName': 'data_format',
        'dlParamName': 'dataFormat',
        'type': 'string',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'DepthwiseConv2d',
    'dlOpName': 'depthwiseConv2d',
    'category': 'convolution',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'input', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'filter', 'type': 'tensor'},
      {'tfParamName': 'strides', 'dlParamName': 'strides', 'type': 'number[]'},
      {'tfParamName': 'padding', 'dlParamName': 'pad', 'type': 'string'}, {
        'tfParamName': 'data_format',
        'dlParamName': 'dataFormat',
        'type': 'string',
        'defaultValue': 'NHWC'
      },
      {
        'tfParamName': 'dilations',
        'dlParamName': 'dilations',
        'type': 'number[]'
      }
    ]
  },
  {
    'tfOpName': 'DepthwiseConv2dNative',
    'dlOpName': 'depthwiseConv2d',
    'category': 'convolution',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'input', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'filter', 'type': 'tensor'},
      {'tfParamName': 'strides', 'dlParamName': 'strides', 'type': 'number[]'},
      {'tfParamName': 'padding', 'dlParamName': 'pad', 'type': 'string'}, {
        'tfParamName': 'data_format',
        'dlParamName': 'dataFormat',
        'type': 'string',
        'defaultValue': 'NHWC'
      },
      {
        'tfParamName': 'dilations',
        'dlParamName': 'dilations',
        'type': 'number[]'
      }
    ]
  }
];
