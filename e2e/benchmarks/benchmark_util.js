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
      // Replace -1 or null in input tensor shape.
      const inputShape = inputNode.shape.map(shapeValue => {
        if (shapeValue == null || shapeValue < 0) {
          return 1;
        } else {
          return shapeValue;
        }
      });

      // Construct the input tensor.
      let inputTensor;
      if (inputNode.dtype === 'float32' || inputNode.dtype === 'int32') {
        inputTensor = tf.randomNormal(inputShape, 0, 1000, inputNode.dtype);
      } else {
        throw new Error(
            `The ${inputNode.dtype} dtype of '${inputNode.name}' input ` +
            `at model.inputs[${inputNodeIndex}] is not supported.`);
      }
      tensorArray.push(inputTensor);
    });

    // Return tensor map for tf.GraphModel.
    if (model instanceof tf.GraphModel) {
      const tensorMap = model.inputNodes.reduce((map, inputName, i) => {
        map[inputName] = tensorArray[i];
        return map;
      }, {});
      return tensorMap;
    }

    return tensorArray;
  } catch (e) {
    // Dispose all input tensors when the input construction is failed.
    tensorArray.forEach(tensor => {
      if (tensor instanceof tf.Tensor) {
        tensor.dispose();
      }
    });
    throw e;
  }
}

async function profileInferenceTimeForModel(model, input, numRuns = 1) {
  let predict;
  if (model instanceof tf.GraphModel) {
    predict = model.executeAsync.bind(model);
  } else if (model instanceof tf.LayersModel) {
    predict = model.predict.bind(model);
  } else {
    throw new Error(
        'Please pass in an instance of tf.GraphModel ' +
        'or tf.LayersModel as the first parameter.');
  }
  return profileInferenceTime(predict, [input], numRuns);
}

async function profileInferenceTime(predict, predictArgs = [], numRuns = 1) {
  if (typeof predict !== 'function') {
    throw new Error(
        'The first parameter should be a function, while ' +
        `a(n) ${typeof predict} is found.`);
  }
  if (!Array.isArray(predictArgs)) {
    predictArgs = [predictArgs];
  }

  const elapsedTimeArray = [];
  for (let i = 0; i < numRuns; i++) {
    let start = performance.now();
    const res = await predict(...predictArgs);
    const inferenceTime = performance.now() - start;

    // The values downloading time will be different for different backends.
    start = performance.now();
    // The prediction can be tf.Tensor|tf.Tensor[]|{[name: string]: tf.Tensor}.
    const value = await downloadValuesFromTensorContainer(res);
    const downloadTime = performance.now() - start;

    tf.dispose(res);
    elapsedTimeArray.push({inferenceTime, downloadTime});
  }
  return elapsedTimeArray;
}

async function downloadValuesFromTensorContainer(tensorContainer) {
  let valueContainer;
  if (tensorContainer instanceof tf.Tensor) {
    valueContainer = await tensorContainer.data();
  } else if (Array.isArray(tensorContainer)) {
    // Start value downloads from all tensors.
    const valuePromiseContainer = tensorContainer.map(async item => {
      if (item instanceof tf.Tensor) {
        return item.data();
      }
      return item;
    });
    // Wait until all values are downloaded.
    valueContainer = await Promise.all(valuePromiseContainer);
  } else if (tensorContainer != null && typeof tensorContainer === 'object') {
    valueContainer = {};
    // Start value downloads from all tensors.
    for (const property in tensorContainer) {
      if (tensorContainer[property] instanceof tf.Tensor) {
        valueContainer[property] = tensorContainer[property].data();
      }
    }
    // Wait until all values are downloaded.
    for (const property in valueContainer) {
      if (valueContainer[property] instanceof Promise) {
        valueContainer[property] = await valueContainer[property];
      }
    }
  }
  return valueContainer;
}
