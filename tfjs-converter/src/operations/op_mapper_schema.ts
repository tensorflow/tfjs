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
    'OpMapper': {
      'type': 'object',
      'properties': {
        'tfOpName': {'type': 'string'},
        'category': {'$ref': '#/definitions/Category'},
        'inputs': {
          'type': 'array',
          'items': {'$ref': '#/definitions/InputParamMapper'}
        },
        'attrs': {
          'type': 'array',
          'items': {'$ref': '#/definitions/AttrParamMapper'}
        },
        'customExecutor': {'$ref': '#/definitions/OpExecutor'}
      },
      'required': ['tfOpName'],
      'additionalProperties': false
    },
    'Category': {
      'type': 'string',
      'enum': [
        'arithmetic', 'basic_math', 'control', 'convolution', 'custom',
        'dynamic', 'evaluation', 'image', 'creation', 'graph', 'logical',
        'matrices', 'normalization', 'reduction', 'slice_join', 'spectral',
        'transformation'
      ]
    },
    'InputParamMapper': {
      'type': 'object',
      'properties': {
        'name': {'type': 'string'},
        'type': {'$ref': '#/definitions/ParamTypes'},
        'defaultValue': {
          'anyOf': [
            {'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}},
            {'type': 'number'}, {'type': 'array', 'items': {'type': 'number'}},
            {'type': 'boolean'},
            {'type': 'array', 'items': {'type': 'boolean'}}
          ]
        },
        'notSupported': {'type': 'boolean'},
        'start': {'type': 'number'},
        'end': {'type': 'number'}
      },
      'required': ['name', 'start', 'type'],
      'additionalProperties': false
    },
    'ParamTypes': {
      'type': 'string',
      'enum': [
        'number', 'string', 'number[]', 'bool', 'shape', 'tensor', 'tensors',
        'dtype'
      ]
    },
    'AttrParamMapper': {
      'type': 'object',
      'properties': {
        'name': {'type': 'string'},
        'type': {'$ref': '#/definitions/ParamTypes'},
        'defaultValue': {
          'anyOf': [
            {'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}},
            {'type': 'number'}, {'type': 'array', 'items': {'type': 'number'}},
            {'type': 'boolean'},
            {'type': 'array', 'items': {'type': 'boolean'}}
          ]
        },
        'notSupported': {'type': 'boolean'},
        'tfName': {'type': 'string'},
        'tfDeprecatedName': {'type': 'string'}
      },
      'required': ['name', 'tfName', 'type'],
      'additionalProperties': false
    },
    'OpExecutor': {'type': 'object', 'additionalProperties': false}
  },
  'items': {'$ref': '#/definitions/OpMapper'},
  'type': 'array'
};
