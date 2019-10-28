/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

// This script edits node_modules/@tensorflow/tfjs-core/dist/tests.js to remove
// tests that are incompatible with react native.

import * as fs from 'fs';
import * as path from 'path';

const testsFilePath = path.resolve(
    __dirname, '../node_modules/@tensorflow/tfjs-core/dist/tests.js');
const fileContents = fs.readFileSync(testsFilePath, 'utf-8');

let newContents = fileContents.replace('require("./worker_node_test");', '');
// disable the version test as we may be testing against a version that is not
// yet depended on by the integration tests.
newContents = newContents.replace('require("./version_test");', '');

fs.writeFileSync(testsFilePath, newContents);
