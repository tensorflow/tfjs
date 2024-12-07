/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 * This file is 2/3 of the test suites for keras v3: create->save->predict.
 * E2E from Keras model -> TFJS model -> Comparsion between two models.
 *
 * This file does below things:
 * - Load the TFJS layers model to runtime.
 * - Load inputs data.
 * - Make inference and store in local files.
 */
const tf = require('@tensorflow/tfjs');
const tfc = require('@tensorflow/tfjs-core');
const tfconverter = require('@tensorflow/tfjs-converter');
const tfl = require('@tensorflow/tfjs-layers');
const tfjsNode = require('@tensorflow/tfjs-node');
const ndarray = require('ndarray');
const fs = require('fs');
const join = require('path').join;
const TEST_DATA_DIR = "keras_to_tfjs_create_save_predict_data";

async function load_file(file_path, model_name) {

  const p = tfjsNode.io.fileSystem(file_path)

  const model = await tf.loadLayersModel("file://" + tfjsNode.io.fileSystem(file_path).path + "/model.json");

  const xsDataPath = join(TEST_DATA_DIR, `${model_name}.xs-data.json`);
  const xsShapesPath = join(TEST_DATA_DIR, `${model_name}.xs-shapes.json`);
  const dataArray = fs.readFileSync(xsDataPath, "utf-8", (_, data) => {
    return JSON.parse(data);
  });
  const shapeArray = fs.readFileSync(xsShapesPath, "utf-8", (_, data) => {
    return JSON.parse(data);
  });

  const parsedShape = JSON.parse(shapeArray);
  const parsedData = JSON.parse(dataArray);
  const zippedArray = zip(parsedData, parsedShape);
  console.log("zipped Array: ", zippedArray);
  const res = [];
  for (let i = 0; i < zippedArray.length; i++) {
    const value = zippedArray[i][0];
    const shape = zippedArray[i][1];
    res.push(ndarray(value, shape));
    // res.push(tfjsNode.tensor(value, shape));
  }
  let z;
  if (res.length == 1) {
    z = res[0];
  } else {
    z = res;
  }
  const tensor = tfjsNode.tensor(z.data, z.shape);
  const ys = model.predict(tensor);

  const resultData = ys.arraySync();
  const resultShape = [ys.shape];
  const ysDataPath = join(TEST_DATA_DIR, `${model_name}.ys-data.json`);
  const ysShapesPath = join(TEST_DATA_DIR, `${model_name}.ys-shapes.json`);
  fs.writeFileSync(ysDataPath, JSON.stringify(resultData));
  fs.writeFileSync(ysShapesPath, JSON.stringify(resultShape));
}

function zip(...arrays) {
  const length = Math.min(...arrays.map(arr => arr.length));
  const result = [];
  for (let i = 0; i < length; i++) {
    const zippedItem = arrays.map(arr => arr[i]);
    result.push(zippedItem);
  }
  return result;
}
console.log(`Using tfjs-core version: ${tfc.version_core}`);
console.log(`Using tfjs-layers version: ${tfl.version_layers}`);
console.log(`Using tfjs-node version: ${JSON.stringify(tfjsNode.version)}`);

(async function () {
  await load_file(join(TEST_DATA_DIR, "mlp"), "mlp");
})();
