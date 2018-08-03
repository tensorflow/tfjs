/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
const fs = require('fs');

const files = process.argv.slice(2);

// Load the TensorFlow.dll and create a .def file for the symbols that need to
// be exported.
const symbols = files.map((file) => fs.readFileSync(file))
                    .join('\n')
                    .split('\n')
                    .map((line) => {
                      var match = /^TF_CAPI_EXPORT.*?\s+(\w+)\s*\(/.exec(line);
                      return match && match[1];
                    })
                    .filter((symbol) => symbol !== null);

process.stdout.write('EXPORTS\n' + symbols.join('\n'));
