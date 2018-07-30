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

export const json = {
  '$schema': 'http://json-schema.org/draft-07/schema#',
  'definitions': {
    'Category': {
      'enum': [
        'arithmetic', 'basic_math', 'control', 'convolution', 'creation',
        'dynamic', 'evaluation', 'image', 'graph', 'logical', 'matrices',
        'normalization', 'reduction', 'slice_join', 'transformation'
      ],
      'type': 'string'
    },
    'OpMapper': {
      'properties': {
        'category': {'$ref': '#/definitions/Category'},
        'dlOpName': {'type': 'string'},
        'params':
            {'items': {'$ref': '#/definitions/ParamMapper'}, 'type': 'array'},
        'tfOpName': {'type': 'string'},
        'unsupportedParams': {'items': {'type': 'string'}, 'type': 'array'}
      },
      'type': 'object'
    },
    'ParamMapper': {
      'properties': {
        'converter': {'type': 'string'},
        'defaultValue': {
          'anyOf': [
            {'items': {'type': 'string'}, 'type': 'array'},
            {'items': {'type': 'number'}, 'type': 'array'},
            {'items': {'type': 'boolean'}, 'type': 'array'},
            {'type': ['string', 'number', 'boolean']}
          ]
        },
        'dlParamName': {'type': 'string'},
        'notSupported': {'type': 'boolean'},
        'tfInputIndex': {'type': 'number'},
        'tfInputParamLength': {'type': 'number'},
        'tfParamName': {'type': 'string'},
        'tfParamNameDeprecated': {'type': 'string'},
        'type': {'$ref': '#/definitions/ParamTypes'}
      },
      'type': 'object'
    },
    'ParamTypes': {
      'enum': [
        'bool', 'dtype', 'number', 'number[]', 'shape', 'string', 'tensor',
        'tensors'
      ],
      'type': 'string'
    }
  },
  'items': {'$ref': '#/definitions/OpMapper'},
  'type': 'array'
};
