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
import {spawnSync, exec} from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as rimraf from 'rimraf';
import {BAZEL_PACKAGES} from '../scripts/bazel_packages';

const parser = new argparse.ArgumentParser();

parser.add_argument('tfjs_package', {
  type: String,
  help: 'tfjs package to build dependencies for',
  nargs: '*',
});

parser.add_argument('--all', {
  action: 'store_true',
  help: 'Build all packages',
});

parser.add_argument('--bazel_options', {
  type: String,
  default: '',
  help: 'Options to pass to Bazel',
});

/**
 * Build bazel dependencies for the packages specified by the repeated argument
 * tfjs_package.
 *
 * @example 'yarn build-deps-for tfjs-react-native' builds all bazel
 * dependencies for @tensorflow/tfjs-react-native.
 *
 * @example 'yarn build-deps-for --all' builds all bazel packages.
 */
async function main() {
  const args = parser.parse_args();
  let packageNames: string[] = args.tfjs_package;

  let targets: string[];
  if (args.all) {
    targets = [...BAZEL_PACKAGES].map(dirToTarget);
  } else {
    const allDeps = packageNames
      .map((name): Iterable<string> => getDeps(name))
      .reduce((a, b) => [...a, ...b], []);
    targets = [...allDeps].map(getDepTarget).filter(v => v);
  }

  if (targets.length > 0) {
    if (process.platform === 'win32') {
      const child = exec('yarn bazel build --color=yes '
        + `${args.bazel_options} ${targets.join(' ')}`);
      await new Promise((resolve, reject) => {
        child.stdout.pipe(process.stdout);
        child.stderr.pipe(process.stderr);
        child.on('exit', code => {
          if (code !== 0) {
            reject(code);
          }
          resolve(code);
        });
      });
    } else {
      // Use spawnSync intead of exec for prettier printing.
      const bazelArgs = ['bazel', 'build'];
      if (args.bazel_options) {
        bazelArgs.push(args.bazel_options);
      }
      bazelArgs.push(...targets);
      spawnSync('yarn', bazelArgs, {stdio:'inherit'});
    }
  }

  const tfjsDir = `${__dirname}/node_modules/@tensorflow`;
  rimraf.sync(tfjsDir);
  fs.mkdirSync(tfjsDir, {recursive: true});

  // Copy all built packages to node_modules. Note that this does not install
  // their dependencies, but that's okay since the node resolution algorithm
  // will find dependencies in the root node_modules folder of the repository.
  for (const pkg of BAZEL_PACKAGES) {
    const pkgPath = path.normalize(
      `${__dirname}/../dist/bin/${pkg}/${pkg}_pkg`);

    if (fs.existsSync(pkgPath)) {
      const newPath = path.join(tfjsDir, pkg);
      copyRecursive(pkgPath, newPath);
    }
  }
  chmodRecursive(tfjsDir);
}

/**
 * Get all dependencies and devDependencies of a tfjs package.
 *
 * @param packageName The package name (without @tensorflow/).
 */
function getDeps(packageName: string): Set<string> {
  const packageJsonPath = path.normalize(
    `${__dirname}/../${packageName}/package.json`);

  const pkg = fs.readFileSync(packageJsonPath).toString('utf8');
  const parsedPkg = JSON.parse(pkg) as {
    dependencies?: Record<string, string>,
    devDependencies?: Record<string, string>,
  };

  const allDeps = new Set<string>();
  function addDepsFrom(record?: Record<string, string>) {
    if (record) {
      for (const key of Object.keys(record)) {
        allDeps.add(key);
      }
    }
  }
  addDepsFrom(parsedPkg.dependencies);
  addDepsFrom(parsedPkg.devDependencies);
  return allDeps;
}

/**
 * Get the bazel target corresponding to a package.json dependency.
 *
 * @param dep The package.json dependency.
 * @returns The bazel target corresponding to the dependency or undefined if no
 * such dependency exists.
 */
function getDepTarget(dep: string): string | undefined {
  const found = dep.match(/@tensorflow\/(.*)/);
  if (found) {
    const name = found[1];
    if (BAZEL_PACKAGES.has(name)) {
      return dirToTarget(name);
    }
  }
  return undefined;
}

/**
 * Get the bazel target from a tfjs package's directory.
 *
 * @param dir The tfjs package's directory.
 * @returns The bazel target to build the tfjs package.
 */
function dirToTarget(dir: string) {
  return `//${dir}:${dir}_pkg`;
}

/**
 * Recursively copy a file or directory.
 *
 * @param src The source directory.
 * @param dest The destination to copy src to.
 */
function copyRecursive(src: string, dest: string) {
  // Avoid 'cp -r', which Windows does not suppport
  const stat = fs.lstatSync(src);
  if (stat.isFile()) {
    fs.copyFileSync(src, dest);
  } else if (stat.isDirectory()) {
    const contents = fs.readdirSync(src);
    fs.mkdirSync(dest);
    for (let name of contents) {
      copyRecursive(path.join(src, name),
                    path.join(dest, name));
    }
  }
}

/**
 * Map a function on all files and directories under a path.
 *
 * @param rootPath The path where the function will be mapped.
 * @param mapFn The function to map on all subpaths of the rootPath.
 */
function mapFiles(rootPath: string, mapFn: (path: string) => void) {
  mapFn(rootPath);
  const stat = fs.lstatSync(rootPath);
  if (stat.isDirectory()) {
    const contents = fs.readdirSync(rootPath);
    for (let subPath of contents) {
      mapFiles(path.join(rootPath, subPath), mapFn);
    }
  }
}

/**
 * Recursively change permissions of files to 775.
 *
 * @param rootPath The path where permissions are changed.
 */
function chmodRecursive(rootPath: string) {
  mapFiles(rootPath, path => fs.chmodSync(path, 0o775));
}

main();
