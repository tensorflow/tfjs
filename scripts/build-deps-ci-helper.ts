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

import * as fs from 'fs';
import * as path from 'path';
import * as shell from 'shelljs';

/**
 * A wrapper around shell.exec for readability.
 * @param cmd The bash command to execute.
 * @returns stdout returned by the executed bash script.
 */
function $(cmd: string) {
  const result = shell.exec(cmd, {silent: false});
  if (result.code > 0) {
    console.log('$', cmd);
    console.log(result.stderr);
    process.exit(1);
  }
  return result.stdout.trim();
}

function getPackageFolders() {
  return process.argv.slice(2);
}

function maybeBuildPackage(folder: string) {
  const distPath = path.resolve(folder, './dist');
  const stat = fs.existsSync(distPath)
  if (stat) {
    console.log(`dist folder for ${folder} already exists. Skipping build`);
  }
  else {
    console.log(`dist folder for ${folder} does not exist. Triggering build`);
    console.log(process.cwd());
    shell.cd(folder)
    console.log(process.cwd());
    $('yarn && yarn build-ci')
    shell.cd('..');
    console.log(process.cwd());
  }
}

console.log('in ts')
console.log(process.cwd());
console.log(getPackageFolders());

const packageFolders = getPackageFolders();
packageFolders.forEach((packageFolder) => {
  maybeBuildPackage(packageFolder);
});
