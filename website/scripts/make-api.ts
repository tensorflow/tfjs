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

/*
 *  To run this file, run this command from the root of repo:
 *      ./node_modules/.bin/ts-node ./website/scripts/make-api.ts
 */

import * as fs from 'fs';
import * as minimist from 'minimist';
import * as mkdirp from 'mkdirp';
import * as mustache from 'mustache';
import * as shell from 'shelljs';
import * as ts from 'typescript';

// tslint:disable-next-line:max-line-length
import {DocHeading, DocMethod, DocMethodParam, Docs, DocSubheading} from '../api/view';

const DOCUMENTATION_DECORATOR = '@doc';
// Mirrors the info argument to @doc in decorators.ts.
interface DocInfo {
  heading: string;
  subheading: string;
  namespace?: string;
}

const LIB_TOPLEVEL_NAMESPACE = 'dl';
const API_TEMPLATE_PATH = './website/api/index.html';
const SRC_ROOT = 'src/';
const PROGRAM_ROOT = SRC_ROOT + 'index.ts';
const GITHUB_ROOT = 'https://github.com/PAIR-code/deeplearnjs/';
const HTML_OUT_DIR = '/tmp/deeplearn-new-website/api/';

mkdirp(HTML_OUT_DIR);

if (!fs.existsSync(PROGRAM_ROOT)) {
  throw new Error(
      `Program root ${PROGRAM_ROOT} does not exist. Please run this script ` +
      `from the root of repository.`);
}

const repoPath = process.cwd();

// Initialize the doc headings so we control sort order.
const docHeadings: DocHeading[] = [
  {
    name: 'Tensors',
    subheadings: [
      {name: 'Creation'}, {name: 'Transformations'},
      {name: 'Slicing and Joining'}
    ]
  },
  {
    name: 'Operations',
    subheadings: [
      {name: 'Arithmetic'}, {name: 'Basic math'}, {name: 'Matrices'},
      {name: 'Convolution'}, {name: 'Reduction'}, {name: 'Normalization'},
      {name: 'Images'}, {name: 'RNN'}, {name: 'Classification'},
      {name: 'Logical'}
    ]
  },
  {name: 'Training', subheadings: [{name: 'Gradients'}]},
  {name: 'Performance', subheadings: [{name: 'Memory'}, {name: 'Timing'}]}
];
const docs: Docs = {
  headings: docHeadings
};

// Use the same compiler options that we use to compile the library here.
const tsconfig = JSON.parse(fs.readFileSync('tsconfig.json', 'utf8'));

console.log(`Parsing AST from program root ${PROGRAM_ROOT}`);
const program = ts.createProgram([PROGRAM_ROOT], tsconfig.compilerOptions);
const checker = program.getTypeChecker();

// Visit all the nodes that are transitively linked from the source root.
for (const sourceFile of program.getSourceFiles()) {
  if (!sourceFile.isDeclarationFile) {
    ts.forEachChild(sourceFile, node => visitNode(node, sourceFile));
  }
}

// Sort the methods by name.
for (let i = 0; i < docHeadings.length; i++) {
  const heading = docHeadings[i];
  for (let j = 0; j < heading.subheadings.length; j++) {
    const subheading = heading.subheadings[j];

    subheading.methods.sort((a, b) => {
      if (a.path < b.path) {
        return -1;
      } else if (a.path > b.path) {
        return 1;
      }
      return 0;
    });
  }
}

// Write the HTML.
const htmlFilePath = HTML_OUT_DIR + 'index.html';
const mustacheTemplate = fs.readFileSync(API_TEMPLATE_PATH, 'utf8');
const html = mustache.render(mustacheTemplate, docs);
fs.writeFileSync(htmlFilePath, html);

const {headingsCount, subheadingsCount, methodCount} = computeStatistics(docs);
console.log(
    `API reference written to ${htmlFilePath}\n` +
    `Found: \n` +
    `  ${docHeadings.length} headings\n` +
    `  ${subheadingsCount} subheadings\n` +
    `  ${methodCount} methods`);

function visitNode(node: ts.Node, sourceFile: ts.SourceFile) {
  if (ts.isMethodDeclaration(node)) {
    if (node.decorators != null) {
      let hasOpdoc = false;
      let headingNames: string[];
      let docInfo: DocInfo;
      node.decorators.map(decorator => {
        const decoratorStr = decorator.getText();
        if (decoratorStr.startsWith(DOCUMENTATION_DECORATOR)) {
          const decoratorConfigStr =
              decoratorStr.substring(DOCUMENTATION_DECORATOR.length);
          docInfo = eval(decoratorConfigStr);

          hasOpdoc = true;
        }
      });

      if (hasOpdoc) {
        const methodName = node.name.getText();

        const docMethod =
            serializeMethod(node, methodName, docInfo, sourceFile);

        // Find the heading.
        let heading: DocHeading;
        for (let i = 0; i < docHeadings.length; i++) {
          if (docHeadings[i].name === docInfo.heading) {
            heading = docHeadings[i];
          }
        }
        if (heading == null) {
          heading = {name: docInfo.heading, subheadings: []};
          docHeadings.push(heading);
        }

        // Find the subheading.
        let subheading: DocSubheading;
        for (let i = 0; i < heading.subheadings.length; i++) {
          if (heading.subheadings[i].name === docInfo.subheading) {
            subheading = heading.subheadings[i];
          }
        }
        if (subheading == null) {
          subheading = {name: docInfo.subheading, methods: []};
          heading.subheadings.push(subheading);
        }
        if (subheading.methods == null) {
          subheading.methods = [];
        }

        subheading.methods.push(docMethod);
      }
    }
  }

  ts.forEachChild(node, node => visitNode(node, sourceFile));
}

function serializeParameter(symbol: ts.Symbol): DocMethodParam {
  return {
    name: symbol.getName(),
    documentation: ts.displayPartsToString(symbol.getDocumentationComment()),
    type: checker.typeToString(
        checker.getTypeOfSymbolAtLocation(symbol, symbol.valueDeclaration!)),
    optional: checker.isOptionalParameter(
        symbol.valueDeclaration as ts.ParameterDeclaration)
  };
}

function serializeMethod(
    node: ts.MethodDeclaration, name: string, docInfo: DocInfo,
    sourceFile: ts.SourceFile): DocMethod {
  const symbol = checker.getSymbolAtLocation(node.name);
  const type =
      checker.getTypeOfSymbolAtLocation(symbol, symbol.valueDeclaration!);
  const signature = type.getCallSignatures()[0];

  if (!sourceFile.fileName.startsWith(repoPath)) {
    throw new Error(
        `Error: source file ${sourceFile.fileName} ` +
        `does not start with srcPath provided ${repoPath}.`);
  }
  // Line numbers are 0-indexed.
  const startLine =
      sourceFile.getLineAndCharacterOfPosition(node.getStart()).line + 1;
  const endLine =
      sourceFile.getLineAndCharacterOfPosition(node.getEnd()).line + 1;
  const fileName = sourceFile.fileName.substring(repoPath.length + '/'.length);
  const displayFilename = fileName.substring(SRC_ROOT.length) + '#' + startLine;

  const githubUrl =
      `${GITHUB_ROOT}blob/master/${fileName}#L${startLine}-L${endLine}`;

  const path = LIB_TOPLEVEL_NAMESPACE + '.' +
      (docInfo.namespace != null ? docInfo.namespace + '.' : '') + name;

  return {
    path,
    parameters: signature.parameters.map(serializeParameter),
    returnType: checker.typeToString(signature.getReturnType()),
    documentation: ts.displayPartsToString(signature.getDocumentationComment()),
    fileName: displayFilename,
    githubUrl
  };
}

function computeStatistics(docs: Docs):
    {headingsCount: number, subheadingsCount: number, methodCount: number} {
  let subheadingsCount = 0;
  let methodCount = 0;
  for (let i = 0; i < docs.headings.length; i++) {
    const heading = docs.headings[i];
    subheadingsCount += heading.subheadings.length;
    for (let j = 0; j < heading.subheadings.length; j++) {
      methodCount += heading.subheadings[j].methods.length;
    }
  }
  return {headingsCount: docs.headings.length, subheadingsCount, methodCount};
}
