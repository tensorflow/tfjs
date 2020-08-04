/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
 * This script generates a mapping of Kernel Names to op names as defined by
 * the converter source code. This allows a couple of things for modular builds
 *   1. From a model.json file we can create imports for the ops the converter
 *      will call.
 *   2. From those ops we could validate that the kernels we add to the modular
 *      build match the names of kernels in model.json (this is not necessary
 *      but is potentially useful for alignment).
 *
 * This can also be used to keep our supported ops list up to date.
 *
 * The approach used is to parse the source code of the converter executors
 * (src/operations/executors) for the following kind pattern.
 *   case 'BiasAdd':
 *   case 'AddV2':
 *   case 'Add': {
 *     return [tfc.add(
 *         (getParamValue('a', node, tensorMap, context) as tfc.Tensor),
 *         getParamValue('b', node, tensorMap, context) as tfc.Tensor)];
 *   }
 *
 * Case matchers represent kernel names and tfc.* represent the tfjs op that is
 * called. This example shows that we need to support fallthrough case
 * statements as well.
 *
 */

import * as argparse from 'argparse';
import * as fs from 'fs';
import {CaseClause, CaseOrDefaultClause, Project, SourceFile, SwitchStatement, SyntaxKind} from 'ts-morph';

const parser = new argparse.ArgumentParser();

parser.addArgument(
    '--out', {help: 'Path to output JSON to create', required: true});

// initialize
const project = new Project({});

function getSwitchStatement(source: SourceFile): SwitchStatement {
  let switchStatement: SwitchStatement;
  source.forEachDescendant((node) => {
    if (node.getKindName() === 'SwitchStatement') {
      switchStatement = node as SwitchStatement;
    }
  });
  return switchStatement;
}

function getKernelMappingForFile(source: SourceFile) {
  const switchStatement = getSwitchStatement(source);
  if (switchStatement === null) {
    throw new Error('No switch statment found in executor');
  }
  const caseClauses = switchStatement.getClauses();

  const kernelsToOp: {[key: string]: string[];} = {};
  let currentClauseGroup: string[] = [];
  caseClauses.forEach((caseClause: CaseOrDefaultClause) => {
    if (caseClause instanceof CaseClause) {
      let kernelName;
      caseClause.forEachChild(clausePart => {
        const kind = clausePart.getKindName();
        if (kind === 'StringLiteral') {
          kernelName = clausePart.getText().replace(/\'/g, '');
          currentClauseGroup.push(kernelName);
        }
        if (kind === 'Block' || kind === 'ReturnStatement') {
          const callExprs =
              clausePart.getDescendantsOfKind(SyntaxKind.CallExpression);
          const tfcCallExprs =
              callExprs.filter(expr => expr.getText().match(/tfc/));
          const tfSymbols: Set<string> = new Set();
          for (const tfcCall of tfcCallExprs) {
            const tfcCallStr = tfcCall.getText();
            const functionCallMatcher = /(tfc\.([\w\.]*)\()/g;
            const matches = tfcCallStr.match(functionCallMatcher);
            if (matches != null && matches.length > 0) {
              for (const match of matches) {
                // extract the method name (and any namespaces used to call it)
                const symbolMatcher = /(tfc\.([\w\.]*)\()/;
                const symbol = match.match(symbolMatcher)[2];
                tfSymbols.add(symbol);
              }
            }
          }
          for (const kern of currentClauseGroup) {
            kernelsToOp[kern] = Array.from(tfSymbols);
          }
          currentClauseGroup = [];
        }
      });
    }
  });

  return kernelsToOp;
}

function getKernelMapping() {
  const sourceFiles = project.getSourceFiles();

  const kernelsToOp: {[key: string]: string[];} = {};

  for (const sourceFile of sourceFiles) {
    const mapping = getKernelMappingForFile(sourceFile);
    Object.assign(kernelsToOp, mapping);
  }
  return kernelsToOp;
}

async function run(outputFilePath: string) {
  const EXECUTORS_PATH = 'src/operations/executors/*_executor.ts';
  project.addSourceFilesAtPaths(EXECUTORS_PATH);

  const kernelMapping = getKernelMapping();

  const pairs: Array<[string, string[]]> = Object.entries(kernelMapping).sort();
  const sortedKernelMapping: {[key: string]: string[];} = {};
  pairs.forEach(([k, v]) => {
    sortedKernelMapping[k] = v;
  });
  const replacer: null = null;
  const space = 2;
  fs.writeFileSync(
      outputFilePath, JSON.stringify(sortedKernelMapping, replacer, space),
      {encoding: 'utf8'});
}

const args = parser.parseArgs();
console.log('Writing output to', args.out);
run(args.out);
