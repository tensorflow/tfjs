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

// tslint:disable-next-line: no-imports-from-dist
import * as tensorflow from '@tensorflow/tfjs-converter/dist/data/compiled_api';
import {io} from '@tensorflow/tfjs-core';

import {getOps} from './model_parser';

const SIMPLE_MODEL: io.ModelArtifacts = {
  format: 'graph-model',
  generatedBy: '0.0.0',
  convertedBy: 'Test Data',
  modelTopology: {
    node: [
      {
        name: 'Input',
        op: 'Placeholder',
        attr: {
          dtype: {
            type: tensorflow.DataType.DT_INT32,
          },
          shape: {shape: {dim: [{size: -1}, {size: 1}]}}
        }
      },
      {name: 'Add1', op: 'Add', input: ['Input', 'Const'], attr: {}},
      {name: 'Sub', op: 'Sub', input: ['Add1', 'Input'], attr: {}}
    ],
    versions: {producer: 1.0, minConsumer: 3}
  }
};

describe('Model parse', () => {
  it('getOps', () => {
    const ops = getOps(SIMPLE_MODEL);
    expect(ops).toEqual(['add', 'sub']);
  });
});
