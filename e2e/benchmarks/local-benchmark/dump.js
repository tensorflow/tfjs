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

const DUMP_LEVEL = {
  CLOSE: 0,
  ANY: 1,
};

function compareData(data1, data2, level = DUMP_LEVEL.CLOSE) {
  if (level == DUMP_LEVEL.CLOSE) {
    let match = true;
    try {
      expectObjectsClose(data1, data2);
    } catch (e) {
      match = false;
    }
    return match;
  } else if (level == DUMP_LEVEL.ANY) {
    return JSON.stringify(data1) === JSON.stringify(data2);
  }
  throw new Error(`Dump level ${level} is not supported!`);
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
    jsonObject[key] = [];
    for (let j = 0; j < tensorsMap[key].length; j++) {
      const data = await (tensorsMap[key][j]).data();
      jsonObject[key].push({
        value: data,
        shape: tensorsMap[key][j].shape,
        dtype: tensorsMap[key][j].dtype
      });
    }
  }
  return jsonObject;
}

async function saveObjectsToFile(jsonObjects, dumpCount, prefix) {
  let newPrefix = '';
  if (prefix !== '') {
    newPrefix = `${prefix.replace(/\//g, '-')}_`;
  }
  const backends = Object.keys(jsonObjects);
  if (dumpCount) {
    for (let i = 0; i < backends.length; i++) {
      const object = jsonObjects[backends[i]];
      const fileName = `${newPrefix}${backends[i]}.json`;
      const a = document.createElement('a');
      const file =
          new Blob([JSON.stringify(object)], {type: 'application/json'});
      a.href = URL.createObjectURL(file);
      a.download = fileName;
      a.click();
      // This log informs tools file has been saved.
      console.log(fileName);
    }
  }
}

async function convertTensorToData(tensor, needInfo = false) {
  const data = await tensor.data();
  const info = {value: data, shape: tensor.shape, dtype: tensor.dtype};
  tensor.dispose();
  if (needInfo) {
    return info;
  }
  return data;
}

async function getPredictionData(output, needInfo = false) {
  if (output instanceof Promise) {
    output = await output;
  }

  if (output instanceof tf.Tensor) {
    output = [await convertTensorToData(output, needInfo)];
  } else if (Array.isArray(output)) {
    for (let i = 0; i < output.length; i++) {
      if (output[i] instanceof tf.Tensor) {
        output[i] = await convertTensorToData(output[i], needInfo);
      }
    }
    return output;
  } else if (output != null && typeof output === 'object') {
    for (const property in output) {
      if (output[property] instanceof tf.Tensor) {
        output[property] =
            await convertTensorToData(output[property], needInfo);
      }
    }
  }
  return output;
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

async function predictAndGetData(
    predict, model, inferenceInput, benchmark, enableDump) {
  const prediction = await predict(model, inferenceInput);
  const graphModel = getGraphModel(model, benchmark);
  let intermediateData = {};
  enableDump = enableDump && !!graphModel;
  if (enableDump) {
    intermediateData =
        await getIntermediateTensorInfo(graphModel.getIntermediateTensors());
  }
  const predictionData = await getPredictionData(prediction);
  if (enableDump) {
    graphModel.disposeIntermediateTensors();
  }
  return {data: predictionData, intermediateData};
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

  const predictObject = await getPredictionData(prediction, true);
  tf.env().set('KEEP_INTERMEDIATE_TENSORS', savedKeepIntermediateTensors);
  return predictObject;
}

/**
 * Dump a single op specified by outputNodeName.
 * @param model The loaded model.
 * @param reference The reference result.
 * @param backend The predict backend.
 * @param outputNodeName Output node name.
 * @param index Used as timestamp.
 */
async function dumpOp(model, reference, backend, outputNodeName, index) {
  const modelJson = model.artifacts;
  const predictObject =
      await predictOp(model, modelJson, reference, outputNodeName, backend);
  if (predictObject) {
    return [{index, ...predictObject}, {index, ...reference[outputNodeName]}];
  }
  return [null, null];
}

/**
 * Dump the predict results of two backends and save diffs to files.
 * @param model The loaded model.
 * @param input The predict results from backends.
 * @param backends [predict backend, reference backend].
 * @param benchmark Used for getting graph model.
 * @param prefix Used for generating dump file name.
 * @param level 0, dump close diffs. 1, dump any diffs.
 * @param length Used for controlling how many tensors will be dumped. -1 dump
 *     all.
 */
async function dump(
    model, input, benchmark = '', prefix = '', level = DUMP_LEVEL.CLOSE,
    length = 1) {
  const backends = Object.keys(input);
  const jsonObject1 = input[backends[0]];
  const jsonObject2 = input[backends[1]];
  const dumpObjects1 = {};
  const dumpObjects2 = {};
  const keys = Object.keys(jsonObject1);
  prefix = `dump_${prefix}_${level}`;
  let dumpCount = 0;
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    if (!compareData(jsonObject1[key], jsonObject2[key], level)) {
      const graphModel = getGraphModel(model, benchmark);
      const [objects1, objects2] =
          await dumpOp(graphModel, jsonObject2, backends[0], key, i);
      if (objects1 && objects2 && !compareData(objects1, objects2, level)) {
        dumpObjects1[key] = objects1;
        dumpObjects2[key] = objects2;
        dumpCount++;
      }
    }
    // Break when diff count equals dumpLength to avoid downloading large file.
    if (length != -1 && dumpCount == length) {
      break;
    }
  }
  const dumpData = {[backends[0]]: dumpObjects1, [backends[1]]: dumpObjects2};
  await saveObjectsToFile(dumpData, dumpCount, prefix);
  if (dumpCount) {
    console.warn(`Total dumped ${dumpCount} item(s).`);
  }
}
