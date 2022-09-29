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

async function getIntermediateTensorInfo(tensorsMap) {
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
    jsonObjects, backends, level, dumpCount, prefix) {
  let newPrefix = '';
  if (prefix !== '') {
    newPrefix = `${prefix.replace(/\//g, '-')}_`;
  }
  if (((level < 2) && dumpCount) || (level === 2)) {
    for (let i = 0; i < jsonObjects.length; i++) {
      const object = jsonObjects[i];
      const fileName = `${newPrefix}${backends[i]}.json`;
      const a = document.createElement('a');
      const file =
          new Blob([JSON.stringify(object)], {type: 'application/json'});
      a.href = URL.createObjectURL(file);
      a.download = fileName;
      a.click();
      await sleep(150);
      // This log informs tools file has been saved.
      console.log(fileName);
    }
  }
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
      await getIntermediateTensorInfo(model.getIntermediateTensors());
  await getPredictionData(prediction);
  model.disposeIntermediateTensors();
  return intermediateData;
}

async function dumpOp(model, dumpedJson, backends, outputNodeName, index) {
  let objects1 = {};
  let objects2 = {};
  const modelJson = model.artifacts;
  const referenceObject = await predictOp(
      model, modelJson, dumpedJson, outputNodeName, backends[1]);

  const predictObject = await predictOp(
      model, modelJson, dumpedJson, outputNodeName, backends[0]);
  if (predictObject && referenceObject) {
    objects1 = {index, ...predictObject[`${outputNodeName}`]};
    objects2 = {index, ...referenceObject[`${outputNodeName}`]};
  }
  return [objects1, objects2];
}


/**
 * Dump the predict results of two backends and save diffs to files.
 * @param model The loaded model.
 * @param jsonObjects The predict results from backends.
 * @param backends [predict backend, reference backend].
 * @param benchmark Used for generating dump file name.
 * @param level 0, dump close diffs. 1, dump any diffs. 2, dump all tensors.
 * @param length Used for controlling how many tensors will be dumped. -1 dump
 *     all.
 */
async function dump(
    model, jsonObjects, backends, benchmark = '', level = 0, length = 1) {
  const jsonObject1 = jsonObjects[0];
  const jsonObject2 = jsonObjects[1];
  const dumpObjects1 = {};
  const dumpObjects2 = {};
  const keys = Object.keys(jsonObjects[0]);
  const prefix = `dump_${benchmark}_${level}`;
  let dumpCount = 0;
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    if (compareData(jsonObject1[key], jsonObject2[key], level)) {
      if (level == 2) {
        // Dump all tensors.
        if (jsonObject1[key]['index']) {
          dumpObjects1[`${key}`] = jsonObject1[key];
          dumpObjects2[`${key}`] = jsonObject2[key];
        } else {
          dumpObjects1[`${key}`] = {index: i, ...jsonObject1[key]};
          dumpObjects2[`${key}`] = {index: i, ...jsonObject2[key]};
        }
      } else {
        // Dump close diff or any diff.
        const graphModel = getGraphModel(model, benchmark);
        const [objects1, objects2] =
            await dumpOp(graphModel, jsonObject2, backends, key, i, level);
        if (compareData(objects1, objects2, level)) {
          dumpObjects1[`${key}`] = objects1;
          dumpObjects2[`${key}`] = objects2;
        }
      }

      dumpCount++;
    }
    // Break when diff count equals dumpLength to avoid downloading large file.
    if (length != -1 && dumpCount == length) {
      break;
    }
  }

  await saveObjectsToFile(
      [dumpObjects1, dumpObjects2], backends, level, dumpCount, prefix);
  if (dumpCount) {
    console.warn(`Total dumped ${dumpCount} item(s).`);
  }
}
