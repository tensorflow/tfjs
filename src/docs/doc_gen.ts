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
import * as tfc from '@tensorflow/tfjs-core';
import * as fs from 'fs';
import fetch from 'node-fetch';

import * as arithmetic from '../operations/op_list/arithmetic.json';
import * as basicMath from '../operations/op_list/basic_math.json';
import * as control from '../operations/op_list/control.json';
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
  ...(control as {}) as OpMapper[], ...(convolution as {}) as OpMapper[],
  ...(creation as {}) as OpMapper[], ...(logical as {}) as OpMapper[],
  ...(image as {}) as OpMapper[], ...(graph as {}) as OpMapper[],
  ...(matrices as {}) as OpMapper[], ...(normalization as {}) as OpMapper[],
  ...(reduction as {}) as OpMapper[], ...(sliceJoin as {}) as OpMapper[],
  ...(transformation as {}) as OpMapper[]
];
const GITHUB_URL_PREFIX =
    'https://raw.githubusercontent.com/tensorflow/tfjs-website';
const CORE_API_PREFIX =
    `/master/source/_data/api/${tfc.version_core}/tfjs-core.json`;

async function genDoc() {
  const response = await fetch(GITHUB_URL_PREFIX + CORE_API_PREFIX);
  const json = await response.json();
  // tslint:disable-next-line:no-any
  const coreApis = json.docs.headings.reduce((list: Array<{}>, h: any) => {
    return h.subheadings ? list.concat(h.subheadings.reduce(
                               // tslint:disable-next-line:no-any
                               (sublist: Array<{}>, sub: any) => {
                                 return sublist.concat(sub.symbols);
                               },
                               [])) :
                           list;
  }, []);
  const output: string[] = [];

  output.push('# Supported Tensorflow Ops\n\n');

  generateTable(
      'Operations', 'Arithmetic', (arithmetic as {}) as OpMapper[], output,
      coreApis);
  generateTable(
      'Operations', 'Basic math', (basicMath as {}) as OpMapper[], output,
      coreApis);
  generateTable(
      'Operations', 'Control Flow', (control as {}) as OpMapper[], output,
      coreApis);
  generateTable(
      'Operations', 'Convolution', (convolution as {}) as OpMapper[], output,
      coreApis);
  generateTable(
      'Tensors', 'Creation', (creation as {}) as OpMapper[], output, coreApis);
  generateTable(
      'Tensorflow', 'Graph', (graph as {}) as OpMapper[], output, coreApis);
  generateTable(
      'Operations', 'Logical', (logical as {}) as OpMapper[], output, coreApis);
  generateTable(
      'Operations', 'Matrices', (matrices as {}) as OpMapper[], output,
      coreApis);
  generateTable(
      'Operations', 'Normalization', (normalization as {}) as OpMapper[],
      output, coreApis);
  generateTable(
      'Operations', 'Images', (image as {}) as OpMapper[], output, coreApis);
  generateTable(
      'Operations', 'Reduction', (reduction as {}) as OpMapper[], output,
      coreApis);
  generateTable(
      'Tensors', 'Slicing and Joining', (sliceJoin as {}) as OpMapper[], output,
      coreApis);
  generateTable(
      'Tensors', 'Transformations', (transformation as {}) as OpMapper[],
      output, coreApis);

  console.log(process.cwd());
  fs.writeFileSync(DOC_DIR + 'supported_ops.md', output.join(''));

  console.log(
      `Supported Ops written to ${DOC_DIR + 'supported_ops.md'}\n` +
      `Found ${opMappers.length} ops\n`);
}

function findCoreOps(heading: string, subHeading: string, coreApis: Array<{}>) {
  return coreApis.filter(
      // tslint:disable-next-line:no-any
      (op: any) => op.docInfo.heading === heading &&
          op.docInfo.subheading === subHeading);
}

function generateTable(
    heading: string, subHeading: string, ops: OpMapper[], output: string[],
    coreApis: Array<{}>) {
  const coreOps = findCoreOps(heading, subHeading, coreApis);
  output.push(`## ${heading} - ${subHeading}\n\n`);
  output.push('|Tensorflow Op Name|Tensorflow.js Op Name|\n');
  output.push('|---|---|\n');
  ops.sort((a, b) => a.tfOpName.localeCompare(b.tfOpName)).forEach(element => {
    output.push(`|${element.tfOpName}|${element.dlOpName}|\n`);
  });

  coreOps
      // tslint:disable-next-line:no-any
      .sort((a: any, b: any) => a.symbolName.localeCompare(b.symbolName))
      // tslint:disable-next-line:no-any
      .forEach((element: any) => {
        if (!ops.find(op => op.dlOpName === element.symbolName)) {
          output.push(`|Not mapped|${element.symbolName}|\n`);
        }
      });
  output.push('\n\n');
}

genDoc();
