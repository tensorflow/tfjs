/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {Tensor} from '../tensor';

function countParams(x: Tensor): number {
  const shape = x.shape;
  if (shape.length > 0) {
    return shape.reduce((a: number, b: number) => a * b);
  } else {
    // Scalar.
    return 1;
  }
}

describeWithFlags('dropout', ALL_ENVS, () => {
  const dropoutLevels = [0, 0.75];
  const seed = 23;
  for (const dropoutLevel of dropoutLevels) {
    it(`Level = ${dropoutLevel}`, async () => {
      const x = tf.range(1, 21).reshape([10, 2]);
      const y = tf.dropout(x, tf.scalar(dropoutLevel), null, seed);
      expect(y.dtype).toEqual(x.dtype);
      expect(y.shape).toEqual(x.shape);
      const xValue = await x.data();
      const yValue = await y.data();
      let nKept = 0;
      for (let i = 0; i < xValue.length; ++i) {
        if (yValue[i] !== 0) {
          nKept++;
          expect(yValue[i]).toBeCloseTo(1 / (1 - dropoutLevel) * xValue[i]);
        }
      }
      const numel = countParams(x);
      if (dropoutLevel === 0) {
        expect(nKept).toEqual(numel);
      } else {
        expect(nKept).toBeLessThan(numel);
      }
    });
  }
});
