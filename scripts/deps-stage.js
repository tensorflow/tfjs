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
const path = require('path');
const util = require('util');

const copy = util.promisify(fs.copyFile);
const rename = util.promisify(fs.rename);
const symlink = util.promisify(fs.symlink);
const {libName, depsLibPath} = require('./deps-constants.js');

const action = process.argv[2];
let targetDir = process.argv[3];

// Some windows machines contain a trailing " char:
if (targetDir != undefined && targetDir.endsWith('"')) {
  targetDir = targetDir.substr(0, targetDir.length - 1);
}

const destLibPath = path.join(targetDir, libName);

/**
 * Symlinks the extracted libtensorflow library to the destination path. If the
 * symlink fails, a copy is made.
 */
async function symlinkDepsLib() {
  if (destLibPath === undefined) {
    throw new Error('Destination path not supplied!');
  }
  try {
    await symlink(depsLibPath, destLibPath);
  } catch (e) {
    console.error(
        `  * Symlink of ${destLibPath} failed, creating a copy on disk.`);
    await copy(depsLibPath, destLibPath);
  }
}

/**
 * Moves the deps library path to the destination path.
 */
async function moveDepsLib() {
  await rename(depsLibPath, destLibPath);
}

/**
 * Symlink or move libtensorflow for building the binding.
 */
async function run(action) {
  if (action.endsWith('symlink')) {
    // Symlink will happen during `node-gyp rebuild`
    await symlinkDepsLib();
  } else if (action.endsWith('move')) {
    // Move action is used when installing this module as a package.
    await moveDepsLib();
  } else {
    throw new Error('Invalid action: ' + action);
  }
}

run(action);
