/**
 * @license
 * Copyright 2022 Google LLC.
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
 * DUMP_LEVEL.BIGDIFF: dumping when difference is greater than the default
 * epsilon. DUMP_LEVEL.ANYDIFF: dumping when difference is greater than 0.
 */
const DUMP_LEVEL = {
  BIGDIFF: 0,
  ANYDIFF: 1,
};

function compareData(data1, data2, level = DUMP_LEVEL.BIGDIFF) {
  let epsilon = level == DUMP_LEVEL.ANYDIFF ? 0 : -1;
  let match = true;
  try {
    expectObjectsClose(data1, data2, epsilon);
  } catch (e) {
    match = false;
  }
  return match;
}

function getGraphModel(model) {
  if (model instanceof tf.GraphModel) {
    return model;
  } else if (model.model instanceof tf.GraphModel) {
    return model.model;
  } else if (
      model.baseModel && model.baseModel.model instanceof tf.GraphModel) {
    return model.baseModel.model;
  } else {
    console.warn(`Model doesn't support dump!`);
    return null;
  }
}

async function getIntermediateTensorInfo(tensorsMap) {
  if (!tensorsMap) {
    return;
  }
  const jsonObject = {};
  const keysOfTensors = Object.keys(tensorsMap);
  for (let i = 0; i < keysOfTensors.length; i++) {
    const key = keysOfTensors[i];
    jsonObject[key] = [];
    for (let j = 0; j < tensorsMap[key].length; j++) {
      if (tensorsMap[key][j] == null) {
        continue;
      }
      // For universal-sentence-encoder, its inputs are disposed by model.
      try {
        const data = await (tensorsMap[key][j]).data();
        jsonObject[key].push({
          value: data,
          shape: tensorsMap[key][j].shape,
          dtype: tensorsMap[key][j].dtype
        });
      } catch (e) {
        console.error(`${keysOfTensors[i]} ` + e.message);
      }
    }
  }
  return jsonObject;
}

async function saveObjectsToFile(jsonObjects, prefix) {
  let newPrefix = '';
  if (prefix !== '') {
    newPrefix = `${prefix.replace(/\//g, '-')}_`;
  }
  const backends = Object.keys(jsonObjects);
  if (Object.keys(jsonObjects[backends[0]]).length == 0) {
    return;
  }
  for (let i = 0; i < backends.length; i++) {
    const object = jsonObjects[backends[i]];
    const fileName = `${newPrefix}${backends[i]}.json`;
    const a = document.createElement('a');
    const file = new Blob([JSON.stringify(object)], {type: 'application/json'});
    a.href = URL.createObjectURL(file);
    a.download = fileName;
    a.click();
    // This log informs tools file has been saved.
    console.log(fileName);
  }
}

/**
 * Create a NamedTensorMap from an output node name.
 * @param outputNodeName Output node name.
 * @param modelJson The parsed model.json.
 * @param dumpedJson The dumped tensor infomation (including shape, dtype,
 *     value).
 *
 * @returns A NamedTensorMap.
 */
async function createNamedTensorMap(outputNodeName, modelJson, dumpedJson) {
  const modelNodes = modelJson['modelTopology']['node'];
  let inputs = [];
  for (let i = 0; i < modelNodes.length; i++) {
    if (outputNodeName === modelNodes[i].name && modelNodes[i].input) {
      inputs = modelNodes[i].input;
      break;
    }
  }
  // In
  // https://storage.googleapis.com/tfhub-tfjs-modules/mediapipe/tfjs-model/face_landmarks_detection/attention_mesh/1/model.json,
  // some inputs are prefixed with '^'.
  if (!inputs || inputs.length == 0 || inputs[0].startsWith('^')) {
    return null;
  }

  let tensorMap = {};
  for (let i = 0; i < inputs.length; i++) {
    const key = inputs[i].split(':')[0];
    if (dumpedJson[key] == null || dumpedJson[key][0] == null) {
      console.warn('Tensor ' + key + ' is null!');
      return null;
    }
    const tensorInfo = dumpedJson[key][0];
    const tensor = tf.tensor(
        Object.values(tensorInfo.value), tensorInfo.shape, tensorInfo.dtype);
    tensorMap[key] = tensor;
  }

  return tensorMap;
}

async function predictOp(
    model, modelJson, dumpedJson, outputNodeName, backend) {
  await tf.setBackend(backend);
  const tensorMap =
      await createNamedTensorMap(outputNodeName, modelJson, dumpedJson);
  if (tensorMap == null) {
    return null;
  }
  let prediction;
  let savedKeepIntermediateTensors;
  try {
    savedKeepIntermediateTensors =
        tf.env().getBool('KEEP_INTERMEDIATE_TENSORS');
    tf.env().set('KEEP_INTERMEDIATE_TENSORS', false);
  } catch (e) {
    console.warn(e.message);
  }
  try {
    // TODO(#6861): Support tensor with type conversion.
    prediction = await model.executeAsync(tensorMap, outputNodeName);
  } catch (e) {
    tf.env().set('KEEP_INTERMEDIATE_TENSORS', savedKeepIntermediateTensors);
    console.warn(e.message);
    return null;
  }

  const predictOpObject = await getPredictionData(prediction, true);
  tf.env().set('KEEP_INTERMEDIATE_TENSORS', savedKeepIntermediateTensors);
  return predictOpObject;
}

/**
 * Dump the predict results of two backends and save diffs to files.
 * @param model The loaded model.
 * @param input The actual and expected results from different backends.
 * @param prefix Used for generating dump file name.
 * @param level 0, dump big diffs. 1, dump any diffs.
 * @param length Used for controlling how many tensors will be dumped. -1 dump
 *     all.
 */
async function dump(
    model, input, prefix = '', level = DUMP_LEVEL.BIGDIFF, length = 1) {
  const graphModel = getGraphModel(model);
  if (graphModel == null || length == 0) {
    return;
  }
  const backends = Object.keys(input);
  const actualObject = input[backends[0]];
  const expectedObject = input[backends[1]];
  const dumpActualObject = {};
  const dumpExpectedObject = {};
  const keys = Object.keys(actualObject);
  prefix = `dump_${prefix}_${level}`;
  let dumpCount = 0;
  const modelJson = graphModel.artifacts;
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    if (compareData(actualObject[key], expectedObject[key], level)) {
      continue;
    }
    const predictOpObject = await predictOp(
        graphModel, modelJson, expectedObject, key, backends[0]);
    const [actualOpObject, expectedOpObject] = predictOpObject ?
        [{...predictOpObject, i}, {...expectedObject[key], i}] :
        [null, null];
    if (compareData(actualOpObject, expectedOpObject, level)) {
      continue;
    }
    if (actualOpObject && expectedOpObject) {
      dumpActualObject[key] = actualOpObject;
      dumpExpectedObject[key] = expectedOpObject;
      dumpCount++;
    }
    // Break when diff count equals dumpLength to avoid downloading large file.
    if (length != -1 && dumpCount == length) {
      break;
    }
  }
  const dumpData =
      {[backends[0]]: dumpActualObject, [backends[1]]: dumpExpectedObject};
  await saveObjectsToFile(dumpData, prefix);
  if (dumpCount) {
    console.log(`Total dumped ${dumpCount} item(s).`);
  }
}
