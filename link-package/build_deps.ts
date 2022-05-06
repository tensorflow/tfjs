/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import * as argparse from 'argparse';
import {spawnSync} from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

const PACKAGES: ReadonlySet<string> = new Set([
  'tfjs-core', 'tfjs-backend-cpu', 'tfjs-backend-webgl', 'tfjs-backend-webgpu',
  'tfjs-converter', 'tfjs-tflite', 'tfjs-layers', 'tfjs-data',
]);

const parser = new argparse.ArgumentParser();

parser.add_argument('tfjs_package', {
  type: String,
  help: 'tfjs package to build dependencies for',
});

function getDepTarget(dep: string): string | undefined {
  const found = dep.match(/@tensorflow\/(.*)/);
  if (found) {
    const name = found[1];
    if (PACKAGES.has(name)) {
      return `//${name}:${name}_pkg`;
    }
  }
  return undefined;
}

async function main() {
  const args = parser.parse_args();
  const packageName: string = args.tfjs_package;

  const packageJsonPath = path.normalize(
    `${__dirname}/../${packageName}/package.json`);

  const pkg = fs.readFileSync(packageJsonPath).toString('utf8');
  const parsedPkg = JSON.parse(pkg) as {
    dependencies?: Record<string, string>,
    devDependencies?: Record<string, string>,
  };

  let allDeps = new Set<string>();
  function addDepsFrom(record?: Record<string, string>) {
    if (record) {
      for (const key of Object.keys(record)) {
        allDeps.add(key);
      }
    }
  }
  addDepsFrom(parsedPkg.dependencies);
  addDepsFrom(parsedPkg.devDependencies);

  const targets = [...allDeps].map(getDepTarget).filter(v => v);

  console.log(`bazel build ${targets.join(' ')}`);
  // Use this intead of exec to preserve colors.
  spawnSync('bazel', ['build', ...targets], {stdio:'inherit'});

  const tfjsDir = `${__dirname}/node_modules/@tensorflow`;
  spawnSync('rm', ['-rf', tfjsDir], {stdio: 'inherit'});
  spawnSync('mkdir', ['-p', tfjsDir], {stdio: 'inherit'});

  for (const pkg of PACKAGES) {
    const pkgPath = path.normalize(
      `${__dirname}/../dist/bin/${pkg}/${pkg}_pkg`);

    if (fs.existsSync(pkgPath)) {
      const newPath = path.join(tfjsDir, pkg);
      spawnSync('cp', ['-r', pkgPath, newPath], {stdio: 'inherit'});
    }
  }
  spawnSync('chmod', ['-R', '+w', tfjsDir], {stdio: 'inherit'});
}


main();
