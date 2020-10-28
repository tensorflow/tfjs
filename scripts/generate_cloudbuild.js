// Copyright 2020 Google LLC. All Rights Reserved.
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

const yaml = require('js-yaml');
const fs = require('fs');
const path = require('path');
const { printTable } = require('console-table-printer');

const DEPENDENCY_GRAPH = JSON.parse(
    fs.readFileSync('scripts/package_dependencies.json'));

// This is a reverse dependencies graph. Each entry in the graph lists the
// packages that depend on it.
const REVERSE_DEPENDENCY_GRAPH = transposeGraph(DEPENDENCY_GRAPH);

/**
 * Transpose a directed graph i.e. reverse the direction of the edges.
 */
function transposeGraph(graph) {
  const transposed = {}
  for (const [nodeName, connectedNodes] of Object.entries(graph)) {
    for (const connectedNode of connectedNodes) {
      if (!transposed[connectedNode]) {
	transposed[connectedNode] = new Set();
      }
      if (!transposed[nodeName]) {
	// Make sure the node itself ends up in the transposed graph.
	transposed[nodeName] = new Set();
      }
      transposed[connectedNode].add(nodeName);
    }
  }
  return transposed;
}

/**
 * Topologically sort a directed acyclic graph.
 *
 * Returns a list of graph nodes such that, by following edges,
 * you can only move forward in the list, not backward.
 */
function topologicalSort(graph) {
  // We can't use a standard sorting algorithm because
  // often, two packages won't have any dependency relationship
  // between each other, meaning they are incomparable.
  const sorted = [];

  while (sorted.length < Object.keys(graph).length) {
    // Find nodes not yet in 'sorted' that have edges
    // only to nodes already in 'sorted'
    const emptyNodes = Object.entries(graph).filter(([node, edges]) => {
      if (sorted.includes(node)) {
	return false;
      }
      for (const edge of edges) {
	if (!sorted.includes(edge)) {
	  return false;
	}
      }
      return true;
    }).map(([node, edges]) => node);

    // If there are no such nodes, then the graph has a cycle.
    if (emptyNodes.length === 0) {
      // TODO(msoulanille): Include the cycle in the error message?
      throw new Error('Dependency graph has a cycle.');
    }

    for (node of emptyNodes) {
      sorted.push(node);
    }
  }
  return sorted;
}

// Topologically sort the dependency tree and arrange
// steps in dependency order.
const DEPENDENCY_ORDER = topologicalSort(DEPENDENCY_GRAPH);

/**
 * Find all subnodes in the subgraph generated by taking the transitive
 * closure at `node`.
 */
function findSubgraph(node, graph, subnodes = new Set()) {
  const directSubnodes = graph[node];
  if (directSubnodes) {
    for (const directSubnode of directSubnodes) {
      if (!subnodes.has(directSubnode)) {
	subnodes.add(directSubnode);
	findSubgraph(directSubnode, graph, subnodes);
      }
    }
  }

  return subnodes;
}

/**
 * Find the transitive closure of dependencies of the given packages.
 */
function findDeps(packages) {
  return new Set(...packages.map(
      packageName => findSubgraph(packageName, DEPENDENCY_GRAPH)));
}

/**
 * Find the reverse dependencies of the given packages, i.e. find the
 * set of packages that include at least one of the given packages in
 * their transitive closure of dependencies.
 */
function findReverseDeps(packages) {
  return new Set(...packages.map(
      packageName => findSubgraph(packageName, REVERSE_DEPENDENCY_GRAPH)));
}

const excludeSteps = new Set(['build-deps', 'yarn-common']);

/**
 * Construct a cloudbuild.yml file that does the following:
 * 1. Builds all the dependencies of `packages`
 * 2. Builds and tests all the packages in `packages`
 * 3. Builds and tests all the reverse dependnecies of `packages`
 */
function generateCloudbuild(packages, writeTo = 'cloudbuild_generated.yml') {
  // Make sure all packages are declared in package_dependencies.json.
  const allPackages = new Set(Object.keys(DEPENDENCY_GRAPH));
  for (const packageName of packages) {
    if (!allPackages.has(packageName)) {
      throw new Error(`Package ${packageName} was not declared in `
		      + 'package_dependencies.json');
    }
  }

  const deps = findDeps(packages);
  const reverseDeps = findReverseDeps(packages);

  toBuild = new Set([...deps, ...packages, ...reverseDeps]);
  toTest = new Set([...packages, ...reverseDeps]);

  // Log what will be built and tested
  const buildTestTable = [];
  for (const packageName of allPackages) {
    buildTestTable.push({
      'Package': packageName,
      'Will Build': toBuild.has(packageName) ? '✔' : '',
      'Will Test': toTest.has(packageName) ? '✔' : ''
    });
  }
  printTable(buildTestTable);

  // Load all the cloudbuild files for the packages
  // that need to be built or tested.
  const packageCloudbuildSteps = new Map();
  for (const packageName of new Set([...toBuild, ...toTest])) {
    const doc = yaml.safeLoad(fs.readFileSync(
	path.join(packageName, 'cloudbuild.yml')));
    packageCloudbuildSteps.set(packageName, new Set(doc.steps));
  }

  // Filter out excluded steps. Also remove test steps if the package is
  // not going to be tested. Change step ids to avoid name conflicts.
  for (const [packageName, steps] of packageCloudbuildSteps.entries()) {
    // TODO(msoulanille): Steps that depend on excluded steps might still
    // need to wait for the steps that the excluded steps wait for.
    for (const step of steps) {
      // Exclude a specific set of steps defined in `excludeSteps`.
      // Only include test steps if the package
      // is to be tested.
      if (excludeSteps.has(step.id) ||
	  (!toTest.has(packageName) && isTestStep(step.id))) {
	steps.delete(step);
	continue;
      }

      // Append package name to each step's id.
      if (step.id) {
	// Test steps are not required to have ids.
	step.id = makeStepId(step.id, packageName);
      }

      // Append package names to step ids in the 'waitFor' field.
      if (step.waitFor) {
	step.waitFor = step.waitFor
	    .filter(id => id && !excludeSteps.has(id))
	    .map(id => makeStepId(id, packageName));
      }
    }
  }

  // Set 'waitFor' fields based on dependencies.
  for (const [packageName, steps] of packageCloudbuildSteps.entries()) {
    // Construct the set of step ids that rules in this package must wait for.
    // All packages depend on 'yarn-common', so we special-case it here.
    const waitForSteps = new Set(['yarn-common']);
    for (const dependencyName of (DEPENDENCY_GRAPH[packageName] || new Set())) {
      const cloudbuildSteps = packageCloudbuildSteps
	    .get(dependencyName) || new Set();

      for (const step of cloudbuildSteps) {
	if (!isTestStep(step.id)) {
	  waitForSteps.add(step.id);
	}
      }
    }

    // Add the above step ids to the `waitFor` field in each step.
    for (const step of steps) {
      step.waitFor = [...new Set([...(step.waitFor || []), ...waitForSteps])]
    }
  }

  // Load the general cloudbuild config
  baseCloudbuild = yaml.safeLoad(
      fs.readFileSync('scripts/cloudbuild_general_config.yml'));

  // Include yarn-common as the first step.
  const steps = [...baseCloudbuild.steps];

  // Arrange steps in dependency order
  const sortedPackageSteps = [];
  for (const packageName of DEPENDENCY_ORDER) {
    const packageSteps = packageCloudbuildSteps.get(packageName);
    if (packageSteps) {
      for (const step of packageSteps) {
	steps.push(step);
      }
    }
  }

  // Remove unused secrets. Cloudbuild fails if extra secrets are included.
  const usedSecrets = new Set();
  for (const step of steps) {
    for (const secret of step.secretEnv || []) {
      usedSecrets.add(secret);
    }
  }
  const secretEnv = baseCloudbuild.secrets[0].secretEnv;
  for (const secret of Object.keys(secretEnv)) {
    if (!usedSecrets.has(secret)){
      delete secretEnv[secret];
    }
  }
  if (Object.keys(secretEnv).length === 0) {
    delete baseCloudbuild.secrets;
  }

  baseCloudbuild.steps = steps;
  fs.writeFileSync(writeTo, yaml.safeDump(baseCloudbuild));
}

function isTestStep(id) {
  // Some steps have no ID, so we assume they're test steps.
  if (! id) {
    return true;
  }
  return id.includes('test');
}

function makeStepId(id, packageName) {
  return `${id}-${packageName}`;
}

exports.generateCloudbuild = generateCloudbuild;
