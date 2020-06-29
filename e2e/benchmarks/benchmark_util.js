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

function generateInput(model) {
  if (model == null) {
    throw new Error('The model does not exist.');
  } else if (model.inputs == null) {
    throw new Error('The model.inputs cannot be found.');
  }

  const tensorArray = [];
  try {
    model.inputs.forEach((inputNode, inputNodeIndex) => {
      // replace -1 or null in input tensor shape
      const inputShape = inputNode.shape.map((shapeValue, dimension) => {
        if (shapeValue == null || shapeValue < 0) {
          return 1;
        } else if (shapeValue == 0) {
          throw new Error(
              `In the model.inputs[${inputNodeIndex}], ` +
              `'${inputNode.name}', the shape[${dimension}] is zero.`);
        } else {
          return shapeValue;
        }
      });

      // construct the input tensor
      let inputTensor;
      if (inputNode.dtype == 'float32' || inputNode.dtype == 'int32') {
        inputTensor = tf.randomNormal(inputShape, 0, 1000, inputNode.dtype);
      } else {
        throw new Error(
            `The ${inputNode.dtype} dtype of '${inputNode.name}' input ` +
            `at model.inputs[${inputNodeIndex}] is not supported.`);
      }
      tensorArray.push(inputTensor);
    });

    // return tensor map for GraphModel
    if (model instanceof tf.GraphModel) {
      if (tensorArray.length !== model.inputNodes.length) {
        throw new Error(
            'The generated input array and model.inputNodes are mismatched. ' +
            `The graph model has ${model.inputNodes.length} input nodes, ` +
            `while the generated input array has ${tensorArray.length} nodes.`);
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
