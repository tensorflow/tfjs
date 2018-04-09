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
import * as fs from 'fs';

import * as arithmetic from '../operations/op_list/arithmetic.json';
import * as basicMath from '../operations/op_list/basic_math.json';
import * as convolution from '../operations/op_list/convolution.json';
import * as creation from '../operations/op_list/creation.json';
import * as graph from '../operations/op_list/graph.json';
import * as image from '../operations/op_list/image.json';
import * as logical from '../operations/op_list/logical.json';
import * as matrices from '../operations/op_list/matrices.json';
import * as normalization from '../operations/op_list/normalization.json';
import * as reduction from '../operations/op_list/reduction.json';
import * as sliceJoin from '../operations/op_list/slice_join.json';
import * as transformation from '../operations/op_list/transformation.json';

import {OpMapper} from '../operations/types';

const DOC_DIR = './docs/';

const opMappers = [
  ...(arithmetic as {}) as OpMapper[], ...(basicMath as {}) as OpMapper[],
  ...(convolution as {}) as OpMapper[], ...(creation as {}) as OpMapper[],
  ...(logical as {}) as OpMapper[], ...(image as {}) as OpMapper[],
  ...(graph as {}) as OpMapper[], ...(matrices as {}) as OpMapper[],
  ...(normalization as {}) as OpMapper[], ...(reduction as {}) as OpMapper[],
  ...(sliceJoin as {}) as OpMapper[], ...(transformation as {}) as OpMapper[]
];

const output: string[] = [];

output.push('# Supported Tensorflow Ops\n\n');

generateTable('Arithmetic', (arithmetic as {}) as OpMapper[], output);
generateTable('Basic Math', (basicMath as {}) as OpMapper[], output);
generateTable('Convolution', (convolution as {}) as OpMapper[], output);
generateTable('Tensor Creation', (creation as {}) as OpMapper[], output);
generateTable('Tensorflow Graph', (graph as {}) as OpMapper[], output);
generateTable('Logical', (logical as {}) as OpMapper[], output);
generateTable('Matrices', (matrices as {}) as OpMapper[], output);
generateTable('Normalization', (normalization as {}) as OpMapper[], output);
generateTable('Image', (image as {}) as OpMapper[], output);
generateTable('Reduction', (reduction as {}) as OpMapper[], output);
generateTable('Slice and Join', (sliceJoin as {}) as OpMapper[], output);
generateTable('Transformation', (transformation as {}) as OpMapper[], output);

console.log(process.cwd());
fs.writeFileSync(DOC_DIR + 'supported_ops.md', output.join(''));

console.log(
    `Supported Ops written to ${DOC_DIR + 'supported_ops.md'}\n` +
    `Found ${opMappers.length} ops\n`);

function generateTable(category: string, ops: OpMapper[], output: string[]) {
  output.push(`## ${category} Ops\n\n`);
  output.push('|Tensorflow Op Name|Tensorflow.js Op Name|\n');
  output.push('|---|---|\n');
  ops.forEach(element => {
    output.push(`|${element.tfOpName}|${element.dlOpName}|\n`);
  });
  output.push('\n\n');
}
