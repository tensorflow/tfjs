

function generateInput(model) {
  if (model == null) {
    throw new Error('The model does not exist');
  } else if (model.inputs == null) {
    throw new Error('The model.inputs cannot be found');
  }

  try {
    const inferenceInputs = [];
    model.inputs.forEach(input => {
      // replace -1 or null in input tensor shape
      const inputShape = [];
      input.shape.forEach(shapeValue => {
        if (shapeValue == null || shapeValue < 0) {
          inputShape.push(1);
        } else if (shapeValue == 0) {
          throw new Error(
              `Warning: In the model.inputs[${inferenceInputIndex}], ` +
              `'${input.name}', shape[${dimension}] is zero`);
        } else {
          inputShape.push(shapeValue);
        }
      });

      // construct the input tensor
      let inputTensor;
      if (input.dtype == 'float32' || input.dtype == 'int32') {
        inputTensor = tf.randomNormal(inputShape, 0, 1, input.dtype);
      } else {
        throw new Error(
            `The ${input.dtype} dtype  of '${input.name}' input ` +
            `at model.inputs[${inferenceInputIndex}] is not supported`);
      }
      inferenceInputs.push(inputTensor);
    });
    return inferenceInputs;
  } catch (e) {
    // dispose input tensors
    for (let tensorIndex = 0; tensorIndex < inferenceInputs.length; tensorIndex++) {
      if (inferenceInputs[tensorIndex] instanceof tf.Tensor) {
        inferenceInputs[tensorIndex].dispose();
      }
    }
    throw e;
  }
}
