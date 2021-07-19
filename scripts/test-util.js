// Copyright 2019 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

const shell = require('shelljs');
const fs = require('fs');

function exec(command, opt, ignoreCode) {
  const res = shell.exec(command, opt);
  if (!ignoreCode && res.code !== 0) {
    shell.echo('command', command, 'returned code', res.code);
    process.exit(1);
  }
  return res;
}

// Construct a dependency graph keyed by dependency package.
// Example:
//   dependencyGraph = {
//     "tfjs-core": ["tfjs-converter", "tfjs", ...],
//     "tfjs": ["tfjs-node"],
//     ...
//   }
function constructDependencyGraph(dependencyFilePath) {
  const str = fs.readFileSync(dependencyFilePath, 'utf8');
  const dependencyInfo = JSON.parse(str);

  const dependencyGraph = {};

  Object.keys(dependencyInfo)
      .forEach(package => dependencyInfo[package].forEach(dependency => {
        if (!dependencyGraph[dependency]) {
          dependencyGraph[dependency] = [];
        }
        dependencyGraph[dependency].push(package);
      }));

  return dependencyGraph;
}

function computeAffectedPackages(dependencyGraph, package) {
  const affectedPackages = new Set();
  traverseDependencyGraph(dependencyGraph, package, affectedPackages);

  return Array.from(affectedPackages);
}

// This function performs a depth-first-search to add affected packages that
// transitively depend on the given package.
function traverseDependencyGraph(graph, package, affectedPackages) {
  // Terminate early if the package has been visited.
  if (affectedPackages.has(package)) {
    return;
  }

  const consumingPackages = graph[package];

  if (!consumingPackages) {
    return;
  }

  consumingPackages.forEach(consumingPackage => {
    traverseDependencyGraph(graph, consumingPackage, affectedPackages);
    affectedPackages.add(consumingPackage);
  });
}

exports.exec = exec;
exports.constructDependencyGraph = constructDependencyGraph;
exports.computeAffectedPackages = computeAffectedPackages;
