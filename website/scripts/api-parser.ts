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

import * as fs from 'fs';
import * as ts from 'typescript';

// tslint:disable-next-line:max-line-length
import {DocClass, DocFunction, DocFunctionParam, DocHeading, Docs, DocSubheading} from '../api/view';

import * as util from './api-util';

const DOCUMENTATION_DECORATOR = '@doc';
const DOCUMENTATION_TYPE_ALIAS = 'docalias';
const DOCUMENTATION_LINK_ALIAS = 'doclink';
const SRC_ROOT = 'src/';
const PROGRAM_ROOT = SRC_ROOT + 'index.ts';

const repoPath = process.cwd();

/**
 * Parses the program.
 */
export function parse():
    {docs: Docs, docLinkAliases: {[symbolName: string]: string}} {
  if (!fs.existsSync(PROGRAM_ROOT)) {
    throw new Error(
        `Program root ${PROGRAM_ROOT} does not exist. Please run this script ` +
        `from the root of repository.`);
  }

  // Initialize the doc headings so we control sort order.
  const docHeadings: DocHeading[] = [
    {
      name: 'Tensors',
      description: '',
      subheadings: [
        {
          name: 'Creation',
          description: `<p>Tensors are the core datastructure of deeplearn.js.
             They are a generalization of vectors and matrices to potentially
             higher dimensions.
             </p>
             <p>We have utility functions for common cases like 1D, 2D, 3D
             and 4D tensors, as well a number of functions to iniatialize
             tensors in ways useful for machine learning</p>`,
          pin: [
            'tensor', 'scalar', 'tensor1d', 'tensor2d', 'tensor3d', 'tensor4d'
          ]
        },
        {name: 'Classes', pin: ['Tensor', 'Variable', 'TensorBuffer']},
        {name: 'Transformations'}, {name: 'Slicing and Joining'}
      ]
    },
    {
      name: 'Operations',
      description: '',
      subheadings: [
        {name: 'Arithmetic', pin: ['add', 'sub', 'mul', 'div']},
        {name: 'Basic math'}, {name: 'Matrices'}, {name: 'Convolution'},
        {name: 'Reduction'}, {name: 'Normalization'}, {name: 'Images'},
        {name: 'RNN'}, {name: 'Logical'}
      ]
    },
    {
      name: 'Training',
      description: '',
      subheadings: [
        {name: 'Gradients'},
        {name: 'Optimizers', pin: ['sgd', 'momentum', 'adagrad', 'adadelta']},
        {name: 'Losses'}, {name: 'Classes'}
      ]
    },
    {
      name: 'Performance',
      description: '',
      subheadings:
          [{name: 'Memory', pin: ['tidy']}, {name: 'Timing', pin: ['time']}]
    },
    {
      name: 'Environment',
      description: '',
      subheadings: [{name: '', pin: ['setBackend']}]
    }
  ];

  // We keep an auxillary map of explicitly marked "subclass" fields on
  // @doc to the method entries
  const subclassMethodMap: {[subclass: string]: DocFunction[]} = {};
  const docTypeAliases: {[type: string]: string} = {};
  const docLinkAliases: {[symbolName: string]: string} = {};

  // Use the same compiler options that we use to compile the library
  // here.
  const tsconfig = JSON.parse(fs.readFileSync('tsconfig.json', 'utf8'));

  console.log(`Parsing AST from program root ${PROGRAM_ROOT}`);
  const program = ts.createProgram([PROGRAM_ROOT], tsconfig.compilerOptions);
  const checker = program.getTypeChecker();

  // Visit all the nodes that are transitively linked from the source
  // root.
  for (const sourceFile of program.getSourceFiles()) {
    if (!sourceFile.isDeclarationFile) {
      ts.forEachChild(
          sourceFile,
          node => visitNode(
              docHeadings, subclassMethodMap, docTypeAliases, docLinkAliases,
              checker, node, sourceFile));
    }
  }

  util.addSubclassMethods(docHeadings, subclassMethodMap);
  util.sortMethods(docHeadings);
  util.replaceDocTypeAliases(docHeadings, docTypeAliases);

  const docs: Docs = {headings: docHeadings};

  return {docs, docLinkAliases};
}

// Visits nodes of the AST, finding documentation annotated with @doc.
function visitNode(
    docHeadings: DocHeading[],
    subclassMethodMap: {[subclass: string]: DocFunction[]},
    docTypeAliases: {[type: string]: string},
    docLinkAliases: {[symbolName: string]: string}, checker: ts.TypeChecker,
    node: ts.Node, sourceFile: ts.SourceFile) {
  if (ts.isMethodDeclaration(node)) {
    const docInfo = util.getDocDecorator(node, DOCUMENTATION_DECORATOR);

    if (docInfo != null) {
      const subheading =
          util.fillHeadingsAndGetSubheading(docInfo, docHeadings);

      const docFunction = serializeMethod(checker, node, docInfo, sourceFile);

      // Static methods are top-level functions,
      if (util.isStatic(node)) {
        subheading.symbols.push(docFunction);
      } else {
        // Non-static methods are class-specific.
        if (docInfo.subclasses != null) {
          for (let i = 0; i < docInfo.subclasses.length; i++) {
            const subclass = docInfo.subclasses[i];
            if (subclassMethodMap[subclass] == null) {
              subclassMethodMap[subclass] = [];
            }
            subclassMethodMap[subclass].push(docFunction);
          }
        }
      }
    }
  } else if (ts.isClassDeclaration(node)) {
    const docLinkAlias = util.getJsdoc(checker, node, DOCUMENTATION_LINK_ALIAS);
    if (docLinkAlias != null) {
      docLinkAliases[node.name.getText()] = docLinkAlias;
    }

    const docInfo = util.getDocDecorator(node, DOCUMENTATION_DECORATOR);
    if (docInfo != null) {
      const subheading =
          util.fillHeadingsAndGetSubheading(docInfo, docHeadings);

      subheading.symbols.push(
          serializeClass(checker, node, docInfo, sourceFile, docHeadings));
    }

    // You can't use both doc link aliases and @doc decorators.
    if (docInfo != null && docLinkAlias != null) {
      throw new Error(
          `Class ${node.name.getText()} has both a ` +
          `doc link alias and a doc decorator.`);
    }
  } else if (
      ts.isInterfaceDeclaration(node) || ts.isTypeAliasDeclaration(node)) {
    const docAlias = util.getJsdoc(checker, node, DOCUMENTATION_TYPE_ALIAS);
    if (docAlias != null) {
      const symbol = checker.getSymbolAtLocation(node.name);
      docTypeAliases[symbol.getName()] = docAlias;
    }
  }

  ts.forEachChild(
      node,
      node => visitNode(
          docHeadings, subclassMethodMap, docTypeAliases, docLinkAliases,
          checker, node, sourceFile));
}

export function serializeClass(
    checker: ts.TypeChecker, node: ts.ClassDeclaration, docInfo: util.DocInfo,
    sourceFile: ts.SourceFile, docHeadings: DocHeading[]): DocClass {
  const symbol = checker.getSymbolAtLocation(node.name);

  const name = symbol.getName();

  const {displayFilename, githubUrl} =
      util.getFileInfo(node, sourceFile, repoPath, SRC_ROOT);
  const docClass: DocClass = {
    symbolName: name,
    namespace: docInfo.namespace,
    documentation: ts.displayPartsToString(symbol.getDocumentationComment()),
    fileName: displayFilename,
    githubUrl,
    methods: [],
    isClass: true
  };

  // Parse the methods that are annotated with @doc.
  node.members.forEach(member => {
    if (ts.isMethodDeclaration(member) && !util.isStatic(member)) {
      const docInfo = util.getDocDecorator(member, DOCUMENTATION_DECORATOR);
      if (docInfo != null) {
        docClass.methods.push(
            serializeMethod(checker, member, docInfo, sourceFile));
      }
    }
  });

  return docClass;
}

export function serializeMethod(
    checker: ts.TypeChecker, node: ts.MethodDeclaration, docInfo: util.DocInfo,
    sourceFile: ts.SourceFile): DocFunction {
  if (!sourceFile.fileName.startsWith(repoPath)) {
    throw new Error(
        `Error: source file ${sourceFile.fileName} ` +
        `does not start with srcPath provided ${repoPath}.`);
  }

  const methodName = node.name.getText();

  const symbol = checker.getSymbolAtLocation(node.name);
  const type =
      checker.getTypeOfSymbolAtLocation(symbol, symbol.valueDeclaration!);
  const signature = type.getCallSignatures()[0];

  const identifierGenericMap = util.getIdentifierGenericMap(node, symbol.name);

  const parameters = signature.parameters.map(
      symbol => serializeParameter(checker, symbol, identifierGenericMap));
  const paramStr = '(' +
      parameters.map(param => param.name + (param.optional ? '?' : ''))
          .join(', ') +
      ')';

  const {displayFilename, githubUrl} =
      util.getFileInfo(node, sourceFile, repoPath, SRC_ROOT);

  // Find a type node in the method signature. This is a return type. If
  // it cannot be found (no return type), fall back to the standard way of
  // getting the type. We do this because getting the full text of the
  // type node is better than using the signature return type.
  let returnType;
  node.forEachChild(child => {
    if (ts.isTypeNode(child)) {
      returnType = child.getText();
    }
  });
  if (returnType == null) {
    // Fall back the the standard way of getting the type, which sometimes
    // gives up and returns 'any' or '{}' for complex types.
    returnType = checker.typeToString(signature.getReturnType());
  }
  returnType = util.sanitizeTypeString(returnType, identifierGenericMap);

  const method: DocFunction = {
    symbolName: symbol.name,
    namespace: docInfo.namespace,
    paramStr,
    parameters,
    returnType,
    documentation: ts.displayPartsToString(signature.getDocumentationComment()),
    fileName: displayFilename,
    githubUrl,
    isFunction: true
  };

  return method;
}

function serializeParameter(
    checker: ts.TypeChecker, symbol: ts.Symbol,
    identifierGenericMap: {[identifier: string]: string}): DocFunctionParam {
  return {
    name: symbol.getName(),
    documentation: ts.displayPartsToString(symbol.getDocumentationComment()),
    type: util.parameterTypeToString(checker, symbol, identifierGenericMap),
    optional: checker.isOptionalParameter(
        symbol.valueDeclaration as ts.ParameterDeclaration)
  };
}
