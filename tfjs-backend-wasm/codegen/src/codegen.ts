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

import {io} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {nodeFileSystemRouter} from '@tensorflow/tfjs-node/dist/io/file_system';
import {writeFileSync} from 'fs';

const HEADER = `
#include "src/cc/backend.h"
#include "src/cc/util.h"

int main() {
`;

const FOOTER = `
}
`;

process.on('unhandledRejection', e => {
  throw e;
});

// Make sure we can recognize file:// urls.
io.registerLoadRouter(nodeFileSystemRouter);

async function main() {
  const handlers =
      io.getLoadHandlers('file://codegen/mobilenet/mobilenet.json');
  const artifacts = await handlers[0].load();
  const weightMap =
      io.decodeWeights(artifacts.weightData, artifacts.weightSpecs);
  let id = 0;
  const lines = [];
  const nameToId: {[name: string]: number} = {};
  for (const name in weightMap) {
    id++;
    nameToId[name] = id;
    const tensor = weightMap[name];
    const bytes = new Uint8Array(tensor.dataSync().buffer);
    const hexCodes: string[] = [];
    for (let i = 0; i < bytes.length; ++i) {
      hexCodes.push(byteToHexCode(bytes[i]));
    }
    lines.push(
        `static const unsigned char weight${id}[] = {${hexCodes.join(',')}};`);
    lines.push(
        `tfjs::wasm::register_tensor(${id}, ${tensor.size}, ` +
        `const_cast<void*>(static_cast<const void*>(weight${id})));`);
  }
  writeFileSync('src/cc/model.cc', HEADER + lines.join('\n') + FOOTER);
}

function byteToHexCode(byte: number): string {
  return '0x' + ('0' + byte.toString(16)).slice(-2);
}

main();
