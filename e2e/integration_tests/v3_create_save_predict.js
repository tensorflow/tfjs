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

/**
 * This file is 2/3 of the test suites for CUJ: create->save->predict.
 *
 * This file does below things:
 * - Create and save models using Layers' API.
 * - Generate random inputs and stored in local file.
 */
const tf = require('@tensorflow/tfjs');
const tfc = require('@tensorflow/tfjs-core');
const tfl = require('@tensorflow/tfjs-layers');
const tfjsNode = require('@tensorflow/tfjs-node');
const fs = require('fs');
const join = require('path').join;
const TEST_DATA_DIR = "test_data_dir";

// process.on('unhandledRejection', ex => {
//   throw ex;
// });

async function load_file(file_path, model_name) {

  const p = tfjsNode.io.fileSystem(file_path)
  console.log("model json ==> ", file_path + "/model.json");
  console.log("Path ==> ", tfjsNode.io.fileSystem(file_path).path);

  const model = await tf.loadLayersModel("file://" + tfjsNode.io.fileSystem(file_path).path + "/model.json");
}


console.log(`Using tfjs-core version: ${tfc.version_core}`);
console.log(`Using tfjs-layers version: ${tfl.version_layers}`);
console.log(`Using tfjs-node version: ${JSON.stringify(tfjsNode.version)}`);

// if (process.argv.length !== 3) {
//   throw new Error('Usage: node tfjs_save.ts <test_data_dir>');
// }
// const testDataDir = process.argv[2];

(async function () {
  await load_file(join(TEST_DATA_DIR, "mlp"), "mlp");
})();

