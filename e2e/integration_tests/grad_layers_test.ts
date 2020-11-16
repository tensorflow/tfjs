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

import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

import * as tfc from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';
import * as tfl from '@tensorflow/tfjs-layers';

import {SMOKE} from './constants';

/**
 *  Tests that tf.grad works for layers models.
 *  Regression test for https://github.com/tensorflow/tfjs/issues/4130
 */
describe(`${SMOKE} tf.grad for layers models`, () => {
  describeWithFlags(`layers_model`, ALL_ENVS, () => {
    let model: tfl.Sequential;

    beforeAll(async () => {
      model = tfl.sequential();
      model.add(tfl.layers.dense({inputShape: [1], units: 1}));
    });

    it(`can compute grad of prediction`, async () => {
      const forward = (x: tfc.Tensor) => model.predict(x) as tfc.Tensor;
      const grad = tfc.grad(forward);

      const input = tfc.tensor([1], [1, 1]);
      const dy = tfc.onesLike(input);
      expect(() => {
        grad(input, dy);
      }).not.toThrow();
    });
  });
});
