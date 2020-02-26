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

function calculateAffectedPackages(dependencyGraph, package) {
  const affectedPackages = new Set();
  traverseDependencyGraph(dependencyGraph, package, affectedPackages);

  return affectedPackages;
}

function constructDependencyGraph(path) {
  try {
    const str = fs.readFileSync(path, 'utf8');
  } catch (err) {
    console.log('Error reading dependency file.');
    return {};
  }

  try {
    const dependencyInfo = JSON.parse(str);
  } catch (err) {
    console.log('Error parsing dependency file to JSON.');
    return {};
  }

  const dependencyGraph = {};

  dependencyInfo.packages.forEach(
      package => {package.dependencies.forEach(dependency => {
        if (!dependencyGraph[dependency]) {
          dependencyGraph[dependency] = [];
        }
        dependencyGraph[dependency].push(package.package);
      })});

  return dependencyGraph;
}

function traverseDependencyGraph(graph, package, affectedPackages) {
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
exports.calculateAffectedPackages = calculateAffectedPackages;
