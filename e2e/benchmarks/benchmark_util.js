

function generateInput(model) {
  if (model == null) {
    throw new Error('The model does not exist');
  } else if (model.inputs == null) {
    throw new Error('The model.inputs cannot be found');
  }

  const tensorArray = [];
  try {
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
      tensorArray.push(inputTensor);
    });

    // return tensor map for GraphModel
    if (model instanceof tf.GraphModel) {
      if (tensorArray.length !== model.inputNodes.length) {
        throw new Error(
            'model.inputs and model.inputNodes are mismatched,' +
            `the graph model has ${model.inputNodes.length} input node, ` +
            `while there are ${tensorArray.length} inputs in model.inputs.`);
      }
      const tensorMap = model.inputNodes.reduce((map, inputName, i) => {
        map[inputName] = tensorArray[i];
        return map;
      }, {});
      return tensorMap;
    }

    return tensorArray;
  } catch (e) {
    // dispose input tensors
    tensorArray.forEach(tensor => {
      if (tensor instanceof tf.Tensor) {
        tensor.dispose();
      }
    });
    throw e;
  }
}
