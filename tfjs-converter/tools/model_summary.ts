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
import * as fs from 'fs';

function summarize(argv: string[]) {
  if (argv.length < 3) {
    console.log('Usage: ts-node pb2json.ts model_file');
    return;
  }

  const sourcePath = process.argv[2];
  console.log('reading pb model file: ' + sourcePath);
  const rawdata = fs.readFileSync(sourcePath);
  const nodes: Array<any> = JSON.parse(rawdata.toString())['modelTopology']['node'];

  const opCount: {[key: string]: number} = {};
  for (const opNode of nodes) {
    let count = 0;
    const op = opNode['op'];
    if (opCount[op]) {
        count = opCount[op];
    } 
    opCount[op] = count + 1; 
  }

  console.log(opCount);
  console.log('Total ops = ' + nodes.length);
}

summarize(process.argv);
