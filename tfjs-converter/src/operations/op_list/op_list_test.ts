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
import * as schema from '../op_mapper_schema';

import * as arithmetic from './arithmetic';
import * as basicMath from './basic_math';
import * as control from './control';
import * as convolution from './convolution';
import * as creation from './creation';
import * as dynamic from './dynamic';
import * as evaluation from './evaluation';
import * as graph from './graph';
import * as image from './image';
import * as logical from './logical';
import * as matrices from './matrices';
import * as normalization from './normalization';
import * as reduction from './reduction';
import * as sliceJoin from './slice_join';
import * as transformation from './transformation';

describe('OpListTest', () => {
  const jsonValidator = new ajv();
  const validator = jsonValidator.compile(schema.json);
  beforeEach(() => {});

  // tslint:disable-next-line:no-any
  const mappersJson: any = {
    arithmetic,
    basicMath,
    control,
    convolution,
    dynamic,
    evaluation,
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
      const valid = validator(mappersJson[key].json);
      if (!valid) console.log(validator.errors);
      expect(valid).toBeTruthy();
    });
  });
});
