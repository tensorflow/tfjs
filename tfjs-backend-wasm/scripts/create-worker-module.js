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
 * This file modifies the Emscripten-generated WASM worker script so that it can
 * be inlined by the tf-backend-wasm bundle.
 */

const fs = require('fs');
const {ArgumentParser} = require('argparse');

const parser = new ArgumentParser();

parser.addArgument('workerFile', {
  type: String,
  help: 'The input worker file to transform.',
});

parser.addArgument('outFile', {
  type: String,
  help: 'The output file path.',
});

parser.addArgument('--cjs', {
  action: 'storeTrue',
  default: false,
  optional: true,
  help: 'Whether to output commonjs instead of esm.',
});

const args = parser.parseArgs();
const workerContents = fs.readFileSync(args.workerFile, "utf8");
const escaped = workerContents.replace(/`/g, '\\`');

if (args.cjs) {
  fs.writeFileSync(`${args.outFile}`,
    `module.exports.wasmWorkerContents = \`${escaped.trim()}\`;`);
} else {
  fs.writeFileSync(`${args.outFile}`,
    `export const wasmWorkerContents = \`${escaped.trim()}\`;`);
}
