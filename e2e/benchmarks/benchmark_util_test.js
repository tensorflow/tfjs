
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
