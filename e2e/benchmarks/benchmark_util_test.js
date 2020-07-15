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

describe('benchmark_util', () => {
  describe('Generate input for model', () => {
    describe('generateInput', () => {
      it('LayersModel', () => {
        const model = tf.sequential(
            {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
        const input = generateInput(model);
        expect(input.length).toEqual(1);
        expect(input[0]).toBeInstanceOf(tf.Tensor);
        expect(input[0].shape).toEqual([1, 3]);
      });
    });
  });

  describe('Profile Memory', () => {
    describe('profileInferenceMemory', () => {
      it('pass in invalid predict', async () => {
        const predict = {};
        await expectAsync(profileInferenceMemory(predict)).toBeRejected();
      });

      it('check tensor leak', async () => {
        const model = tf.sequential(
            {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
        const input = tf.zeros([1, 3]);

        const tensorsBefore = tf.memory().numTensors;
        await profileInferenceMemory(() => model.predict(input));
        expect(tf.memory().numTensors).toEqual(tensorsBefore);

        model.dispose();
        input.dispose();
      });
    });
  });

  describe('Wrap predict function', () => {
    it('graph model with async ops', async () => {
      const modelUrl =
          'https://storage.googleapis.com/tfjs-models/savedmodel/ssd_mobilenet_v1/model.json';
      const model = await tf.loadGraphModel(modelUrl);
      const input = generateInput(model);
      const oldTensorNum = tf.memory().numTensors;
      spyOn(model, 'execute').and.callThrough();
      spyOn(model, 'executeAsync');

      const wrappedPredict = wrapPredictFnForModel(model, input);
      expect(tf.memory().numTensors).toBe(oldTensorNum);
      expect(model.execute.calls.count()).toBe(1);
      expect(model.execute.calls.first().args).toEqual([input]);

      wrappedPredict();
      expect(model.execute.calls.count()).toBe(1);
      expect(model.executeAsync.calls.count()).toBe(1);
      expect(model.executeAsync.calls.first().args).toEqual([input]);

      tf.dispose(input);
      model.dispose();
    });

    it('graph model without async ops', async () => {
      const modelUrl =
          'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json';
      const model = await tf.loadGraphModel(modelUrl);
      const input = tf.zeros([1, 224, 224, 3]);
      const oldTensorNum = tf.memory().numTensors;
      spyOn(model, 'execute').and.callThrough();

      const wrappedPredict = wrapPredictFnForModel(model, input);
      expect(tf.memory().numTensors).toBe(oldTensorNum);
      expect(model.execute.calls.count()).toBe(1);
      expect(model.execute.calls.first().args).toEqual([input]);

      wrappedPredict();
      expect(model.execute.calls.count()).toBe(2);
      expect(model.execute.calls.argsFor(1)).toEqual([input]);

      tf.dispose(input);
      model.dispose();
    });

    it('layers model', () => {
      const model = tf.sequential(
          {layers: [tf.layers.dense({units: 1, inputShape: [10]})]});
      const input = tf.ones([8, 10]);
      spyOn(model, 'predict');

      const wrappedPredict = wrapPredictFnForModel(model, input);
      wrappedPredict();

      expect(model.predict.calls.count()).toBe(1);
      expect(model.predict.calls.first().args).toEqual([input]);

      tf.dispose(input);
      model.dispose();
    });

    it('a model that is neither layers nor graph model', () => {
      const model = {};
      const input = [];
      expect(() => wrapPredictFnForModel(model, input)).toThrowError(Error);
    });
  });
});
