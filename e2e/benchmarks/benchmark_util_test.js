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

/**
 * The unit tests in this file can be run by opening `SpecRunner.html` in
 * browser.
 */

describe('benchmark_util', function() {
  describe('generateInput', function() {
    it('LayersModel', function() {
      const model = tf.sequential(
          {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
      const input = generateInput(model);
      expect(input.length).toEqual(1);
      expect(input[0]).toBeInstanceOf(tf.Tensor);
      expect(input[0].shape).toEqual([1, 3]);
    });
  });
});
