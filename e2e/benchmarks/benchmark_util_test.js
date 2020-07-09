
describe('benchmark_util', function() {
  describe('Generate input for model', function() {
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

  describe('Profile Memory', function() {
    describe('profileInferenceMemory', function() {
      it('pass in invalid predict', async function() {
        const predict = {};
        await expectAsync(profileInferenceMemory(predict)).toBeRejected();
      });

      it('check tensor leak', async function() {
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
});
