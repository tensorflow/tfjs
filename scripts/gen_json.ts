/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as fs from 'fs';
import * as path from 'path';

/**
 * Converts the ts files in src/operations/op_list/* to json files and stores
 * them in python/tensorflowjs/op_list/. These are then used by the python
 * converter.
 */

// Make the directory python/tensorflowjs/op_list/ if it doesn't exist.
const destDir = './python/tensorflowjs/op_list/';
if (!fs.existsSync(destDir)) {
  fs.mkdirSync(destDir);
}

// Go over all .ts files in src/operations/op_list and convert them to json.
const srcDir = './src/operations/op_list';
const fileNames = fs.readdirSync(srcDir);
fileNames.forEach(fileName => {
  const srcPath = path.join(srcDir, fileName);
  try {
    const m = require('../' + srcPath);
    if (m.json === null) {
      console.log('Ignored', srcPath);
      return;
    }
    const destPath = path.join(destDir, fileName.replace('.ts', '.json'));
    fs.writeFileSync(destPath, JSON.stringify(m.json, null, 2));
    console.log('Generated', destPath);
  } catch (ex) {
    console.log('Ignored', srcPath);
  }
});
console.log('Done!');
