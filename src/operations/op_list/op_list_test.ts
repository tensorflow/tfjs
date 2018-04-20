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
import * as ajv from 'ajv';

import * as schema from '../op_mapper_schema.json';

import * as arithmetic from './arithmetic.json';
import * as basicMath from './basic_math.json';
import * as convolution from './convolution.json';
import * as creation from './creation.json';
import * as graph from './graph.json';
import * as image from './image.json';
import * as logical from './logical.json';
import * as matrices from './matrices.json';
import * as normalization from './normalization.json';
import * as reduction from './reduction.json';
import * as sliceJoin from './slice_join.json';
import * as transformation from './transformation.json';

describe('OpListTest', () => {
  const jsonValidator = new ajv();
  const validator = jsonValidator.compile(schema);
  beforeEach(() => {});

  describe('validate schema', () => {
    // tslint:disable-next-line:no-any
    const mappersJson: any = {
      arithmetic,
      basicMath,
      convolution,
      creation,
      logical,
      image,
      graph,
      matrices,
      normalization,
      reduction,
      sliceJoin,
      transformation
    };
    Object.keys(mappersJson).forEach(key => {
      it('should satisfy the schema: ' + key, () => {
        const valid = validator(mappersJson[key]);
        if (!valid) console.log(validator.errors);
        expect(valid).toBeTruthy();
      });
    });
  });
});
