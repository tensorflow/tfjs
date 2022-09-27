/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

function compareData(data1, data2, level = 0) {
  if (level == 0) {
    let notMatch = false;
    try {
      expectObjectsClose(data1, data2);
    } catch (e) {
      notMatch = true;
    }
    return notMatch;
  } else if (level == 1) {
    return JSON.stringify(data1) !== JSON.stringify(data2);
  } else if (level == 2) {
    return true;
  }
}

function getGraphModel(model, benchmark) {
  let graphModel = null;
  if (benchmark === 'bodypix' || benchmark === 'posenet') {
    graphModel = model.baseModel.model;
  } else if (
      benchmark === 'USE - batchsize 30' || benchmark === 'USE - batchsize 1') {
    graphModel = model.model;
  } else {
    graphModel = model;
  }

  if (graphModel instanceof tf.GraphModel) {
    return graphModel;
  }
  console.warn(`Model ${benchmark} doesn't support dump op!`);
  return null;
}

async function getIntermediateTensorsData(tensorsMap) {
  if (!tensorsMap) {
    return;
  }
  const jsonObject = {};
  const keysOfTensors = Object.keys(tensorsMap);
  for (let i = 0; i < keysOfTensors.length; i++) {
    const key = keysOfTensors[i];
    const dataArray = [];
    jsonObject[key] = [];
    for (let j = 0; j < tensorsMap[key].length; j++) {
      const data = await (tensorsMap[key][j]).data();
      dataArray.push(data);
      jsonObject[key].push({
        value: dataArray,
        shape: tensorsMap[key][j].shape,
        dtype: tensorsMap[key][j].dtype
      });
    }
  }
  return jsonObject;
}

async function saveObjectsToFile(
    jsonObjects, backends, level, diffCount, prefix) {
  const backend1 = backends[0];
  const backend2 = backends[1];
  let newPrefix = '';
  if (prefix !== '') {
    newPrefix = `${prefix.replace(/\//g, '-')}_`;
  }
  if (((level < 2) && diffCount) || (level === 2)) {
    const fileNames =
        [`${newPrefix}${backend1}.json`, `${newPrefix}${backend2}.json`];
    for (let i = 0; i < jsonObjects.length; i++) {
      const object = jsonObjects[i];
      const fileName = fileNames[i];
      const a = document.createElement('a');
      const file =
          new Blob([JSON.stringify(object)], {type: 'application/json'});
      a.href = URL.createObjectURL(file);
      a.download = fileName;
      a.click();
      console.log(fileName);
      await sleep(150);
    }
  }
}

// level: 0, dump default diffs. 1, dump any diffs. 2, dump all.
async function compareAndDownload(
    jsonObjects, backends, level, length, keys = [''], prefix = '',
    download = true) {
  const jsonObject1 = jsonObjects[0];
  const jsonObject2 = jsonObjects[1];
  let diffCount = 0;
  const diffObjects1 = {};
  const diffObjects2 = {};
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    if (compareData(jsonObject1[key], jsonObject2[key], level)) {
      if (jsonObject1[key]['index']) {
        diffObjects1[`${key}`] = jsonObject1[key];
        diffObjects2[`${key}`] = jsonObject2[key];
      } else {
        diffObjects1[`${key}`] = {index: i, ...jsonObject1[key]};
        diffObjects2[`${key}`] = {index: i, ...jsonObject2[key]};
      }
      diffCount++;
    }
    // Break when diff count equals dumpLength to avoid downloading large file.
    if (length != -1 && diffCount == length) {
      break;
    }
  }
  if (keys.length === 1) {
    prefix += '_' + keys[0];
  }
  if (download) {
    await saveObjectsToFile(
        [diffObjects1, diffObjects2], backends, level, diffCount, prefix);
  }
  if (diffCount) {
    console.error('Total diff items: ' + diffCount);
  }
  return diffObjects1;
}

/**
 * Create a NamedTensorMap from an output node name.
 * @param outputNodeName output node name.
 * @param modelJson The parsed model.json.
 * @param dumpedJson The dumped tensor infomation (including shape, dtype,
 *     value).
 *
 * @returns A NamedTensorMap.
 */
async function createTensorMap(outputNodeName, modelJson, dumpedJson) {
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
        Object.values(tensorInfo.value[0]), tensorInfo.shape, tensorInfo.dtype);
    tensorMap[key] = tensor;
  }

  return tensorMap;
}

async function predictOp(
    model, modelJson, dumpedJson, outputNodeName, backend) {
  await tf.setBackend(backend);
  const tensorMap =
      await createTensorMap(outputNodeName, modelJson, dumpedJson);
  if (tensorMap == null) {
    return null;
  }
  let prediction;
  try {
    // TODO(#6861): Support tensor with type conversion.
    prediction = await model.executeAsync(tensorMap, outputNodeName);
  } catch (e) {
    console.warn(e.message);
    return null;
  }

  let intermediateData =
      await getIntermediateTensorsData(model.getIntermediateTensors());
  await getPredictionData(prediction);
  model.disposeIntermediateTensors();
  return intermediateData;
}

async function dumpOp(
    model, modelJson, dumpedJson, backends, outputNodeNames, level,
    prefix = '') {
  const predictBackend = backends[0];
  const referenceBackend = backends[1];
  const objects1 = {};
  const objects2 = {};
  for (let i = 0; i < outputNodeNames.length; i++) {
    const outputNodeName = outputNodeNames[i];
    const opReferenceIntermediateObject = await predictOp(
        model, modelJson, dumpedJson, outputNodeName, referenceBackend);

    const opPredictionIntermediateObject = await predictOp(
        model, modelJson, dumpedJson, outputNodeName, predictBackend);
    if (opPredictionIntermediateObject && opReferenceIntermediateObject) {
      objects1[`${outputNodeName}`] = {
        index: i,
        ...opPredictionIntermediateObject[`${outputNodeName}`]
      };
      objects2[`${outputNodeName}`] = {
        index: i,
        ...opReferenceIntermediateObject[`${outputNodeName}`]
      };
    }
  }
  // Op dump only handles diff, change the level to 1.
  level = level == 2 ? 1 : level;
  await compareAndDownload(
      [objects1, objects2], backends, level, -1, Object.keys(objects1), prefix);
}

async function dumpDiff(
    model, objectsDiff, dumpedJson, backends, benchmark, level) {
  if (model == null || model.artifacts == null) {
    console.warn(`${benchmark} doesn't support dump op!`);
    return;
  }

  const keys = Object.keys(objectsDiff);
  if (keys.length === 0) {
    console.warn('Ops have no diff!');
    return;
  }

  const modelJson = model.artifacts;

  await dumpOp(
      model, modelJson, dumpedJson, backends, keys, level,
      `dumpops_${benchmark}_${level}`);
}

async function dump(
    model, jsonObjects, backends, benchmark, level = 0, length = -1) {
  const objectsDiff = await compareAndDownload(
      jsonObjects, backends, level, length, Object.keys(jsonObjects[0]),
      `dumpmodel_${benchmark}_${level}`, true);

  const graphModel = getGraphModel(model, benchmark);
  await dumpDiff(
      graphModel, objectsDiff, jsonObjects[1], backends, benchmark, level);
}
