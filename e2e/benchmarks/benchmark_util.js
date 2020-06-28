

function generateInput(model) {
  const inferenceInputs = [];
  try {
    for (let inferenceInputIndex = 0; inferenceInputIndex < model.inputs.length; inferenceInputIndex++) {
      // construct the input tensor shape
      const inferenceInput = model.inputs[inferenceInputIndex];
      const inputShape = [];
      for (let dimension = 0; dimension < inferenceInput.shape.length; dimension++) {
        const shapeValue = inferenceInput.shape[dimension];
        if (shapeValue == null || shapeValue < 0) {
          inputShape.push(1);
        } else if (shapeValue == 0) {
          await showMsg('Warning: one dimension of an input tensor is zero');
          inputShape.push(shapeValue);
        } else {
          inputShape.push(shapeValue);
        }
      }

      // construct the input tensor
      let inputTensor;
      if (inferenceInput.dtype == 'float32' || inferenceInput.dtype == 'int32') {
        inputTensor = tf.randomNormal(inputShape, 0, 1, inferenceInput.dtype);
      } else {
        throw new Error(
            `The ${inferenceInput.dtype} dtype  of '${inferenceInput.name}' input ` +
            `at model.inputs[${inferenceInputIndex}] is not supported`);
      }
      inferenceInputs.push(inputTensor);
    }

    // run prediction
    let resultTensor;
    if (model instanceof tf.GraphModel && model.executeAsync != null) {
      resultTensor = await model.executeAsync(inferenceInputs);
    } else if (model.predict != null) {
      resultTensor = model.predict(inferenceInputs);
    } else {
      throw new Error("Predict function was not found");
    }
    return resultTensor;
  } finally {
    // dispose input tensors
    for (let tensorIndex = 0; tensorIndex < inferenceInputs.length; tensorIndex++) {
      if (inferenceInputs[tensorIndex] instanceof tf.Tensor) {
        inferenceInputs[tensorIndex].dispose();
      }
    }
  }
}
