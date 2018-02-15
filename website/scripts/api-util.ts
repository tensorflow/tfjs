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
import * as ts from 'typescript';

// tslint:disable-next-line:max-line-length
import {DocClass, DocFunction, DocHeading, Docs, DocSubheading} from '../api/view';

const GITHUB_ROOT = 'https://github.com/PAIR-code/deeplearnjs/';

// Mirrors the info argument to @doc in decorators.ts.
export interface DocInfo {
  heading: string;
  subheading: string;
  namespace?: string;
  subclasses?: string[];
}

export function getDocDecorator(node: ts.Node, decoratorName: string): DocInfo {
  let docInfo: DocInfo;
  if (node.decorators != null) {
    docInfo = parseDocDecorators(node.decorators, decoratorName);
  }
  return docInfo;
}

/**
 * Parses the @doc annotation and returns the typed DocInfo object.
 */
export function parseDocDecorators(
    decorators: ts.NodeArray<ts.Decorator>, decoratorName: string): DocInfo {
  let docInfo: DocInfo = null;
  decorators.map(decorator => {
    const decoratorStr = decorator.getText();
    if (decoratorStr.startsWith(decoratorName)) {
      const decoratorConfigStr = decoratorStr.substring(decoratorName.length);
      docInfo = eval(decoratorConfigStr);
      if (docInfo.subheading == null) {
        docInfo.subheading = '';
      }
    }
  });
  return docInfo;
}

export function addSubclassMethods(
    docHeadings: DocHeading[],
    subclassMethodMap: {[subclass: string]: DocFunction[]}) {
  const subclasses = Object.keys(subclassMethodMap);
  subclasses.forEach(subclass => {
    const methods = subclassMethodMap[subclass];
    // Find the class.
    for (let i = 0; i < docHeadings.length; i++) {
      const heading = docHeadings[i];
      for (let j = 0; j < heading.subheadings.length; j++) {
        const subheading = heading.subheadings[j];
        if (subheading.symbols == null) {
          throw new Error(
              `Subheading '${subheading.name}' has no symbols. ` +
              `Please remove it from the predefined docHeadings, or ` +
              `add methods to the subheading in the code with @doc.`);
        }

        for (let k = 0; k < subheading.symbols.length; k++) {
          const symbol = subheading.symbols[k];
          if (symbol['isClass'] != null && symbol.symbolName === subclass) {
            const classSymbol = symbol as DocClass;
            methods.forEach(method => classSymbol.methods.push(method));
          }
        }
      }
    }
  });
}

// Parse the file info, github URL and filename from a node.
export function getFileInfo(
    node: ts.Node, sourceFile: ts.SourceFile, repoPath: string,
    srcRoot: string): {displayFilename: string, githubUrl: string} {
  // Line numbers are 0-indexed.
  const startLine =
      sourceFile.getLineAndCharacterOfPosition(node.getStart()).line + 1;
  const endLine =
      sourceFile.getLineAndCharacterOfPosition(node.getEnd()).line + 1;
  const fileName = sourceFile.fileName.substring(repoPath.length + '/'.length);
  const displayFilename = fileName.substring(srcRoot.length) + '#' + startLine;

  const githubUrl =
      `${GITHUB_ROOT}blob/master/${fileName}#L${startLine}-L${endLine}`;
  return {displayFilename, githubUrl};
}

// Given a newly seen docInfo from a @doc annotation, fill in headings /
// subheadings and return the subheading.
export function fillHeadingsAndGetSubheading(
    docInfo: DocInfo, docHeadings: DocHeading[]): DocSubheading {
  // Find the heading.
  let heading: DocHeading;
  for (let i = 0; i < docHeadings.length; i++) {
    if (docHeadings[i].name === docInfo.heading) {
      heading = docHeadings[i];
    }
  }
  if (heading == null) {
    heading = {name: docInfo.heading, description: '', subheadings: []};
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
    subheading = {name: docInfo.subheading, symbols: []};
    heading.subheadings.push(subheading);
  }
  if (subheading.symbols == null) {
    subheading.symbols = [];
  }
  return subheading;
}

export function computeStatistics(docs: Docs):
    {headingsCount: number, subheadingsCount: number, methodCount: number} {
  let subheadingsCount = 0;
  let methodCount = 0;
  for (let i = 0; i < docs.headings.length; i++) {
    const heading = docs.headings[i];
    subheadingsCount += heading.subheadings.length;
    for (let j = 0; j < heading.subheadings.length; j++) {
      methodCount += heading.subheadings[j].symbols.length;
    }
  }
  return {headingsCount: docs.headings.length, subheadingsCount, methodCount};
}

// Sorts the doc headings.
export function sortMethods(docHeadings: DocHeading[]) {
  // Sort the methods by name.
  for (let i = 0; i < docHeadings.length; i++) {
    const heading = docHeadings[i];
    for (let j = 0; j < heading.subheadings.length; j++) {
      const subheading = heading.subheadings[j];

      // Pin the symbols in order of the pins.
      const pinnedSymbols = [];
      if (subheading.pin != null) {
        subheading.pin.forEach(pinnedSymbolName => {
          // Loop backwards so we remove symbols.
          for (let k = subheading.symbols.length - 1; k >= 0; k--) {
            const symbol = subheading.symbols[k];
            if (symbol.symbolName === pinnedSymbolName) {
              pinnedSymbols.push(symbol);
              subheading.symbols.splice(k, 1);
            }
          }
        });
      }

      // Sort non-pinned symbols by name.
      subheading.symbols.sort((a, b) => {
        if (a.symbolName < b.symbolName) {
          return -1;
        } else if (a.symbolName > b.symbolName) {
          return 1;
        }
        return 0;
      });

      subheading.symbols = pinnedSymbols.concat(subheading.symbols);
    }
  }
}

export function kind(node: ts.Node): string {
  const keys = Object.keys(ts.SyntaxKind);
  for (let i = 0; i < keys.length; i++) {
    if (ts.SyntaxKind[keys[i]] === node.kind) {
      return keys[i];
    }
  }
}

export function isStatic(node: ts.MethodDeclaration): boolean {
  let isStatic = false;
  node.forEachChild(child => {
    if (child.kind === ts.SyntaxKind.StaticKeyword) {
      isStatic = true;
    }
  });
  return isStatic;
}

/**
 * Finds a jsdoc tag by a given tag name for a symbol. e.g. @docalias number[]
 * => number[].
 */
export function getJsdoc(
    checker: ts.TypeChecker,
    node: ts.InterfaceDeclaration|ts.TypeAliasDeclaration|ts.ClassDeclaration,
    tag: string) {
  const symbol = checker.getSymbolAtLocation(node.name);
  const docs = symbol.getDocumentationComment();
  const tags = symbol.getJsDocTags();
  for (let i = 0; i < tags.length; i++) {
    const jsdocTag = tags[i];
    if (jsdocTag.name === tag) {
      return jsdocTag.text.trim();
    }
  }
}

/**
 * Converts a function parameter symbol to its string type value.
 */
export function parameterTypeToString(
    checker: ts.TypeChecker, symbol: ts.Symbol,
    identifierGenericMap: {[identifier: string]: string}): string {
  const valueDeclaration = symbol.valueDeclaration;

  // Look for type nodes that aren't null and get the full text of the type
  // node, falling back to using the checker to serialize the type.
  let typeStr;
  symbol.valueDeclaration.forEachChild(child => {
    if (ts.isTypeNode(child) && child.kind !== ts.SyntaxKind.NullKeyword) {
      typeStr = child.getText();
    }
  });
  if (typeStr == null) {
    // Fall back to using the checkers method for converting the type to a
    // string.
    typeStr = checker.typeToString(
        checker.getTypeOfSymbolAtLocation(symbol, symbol.valueDeclaration!))
  }

  return sanitizeTypeString(typeStr, identifierGenericMap);
}

/**
 * Sanitizes a type string by removing generics and replacing generics.
 *   e.g. Tensor<R> => Tensor
 *   e.g. T => Tensor
 */
export function sanitizeTypeString(
    typeString: string, identifierGenericMap: {[identifier: string]: string}) {
  // If the return type is part of the generic map, use the mapped
  // type. For example, <T extends Tensor> will replace "T" with
  // "Tensor".
  Object.keys(identifierGenericMap).forEach(identifier => {
    const re = new RegExp('\\b' + identifier + '\\b', 'g');
    typeString = typeString.replace(re, identifierGenericMap[identifier]);
  });

  // Remove generics.
  typeString = typeString.replace(/(<.*>)/, '');

  return typeString;
}

/**
 * Computes a mapping of identifier to their generic type. For example:
 *   method<T extends Tensor>() {}
 * In this example, this method will return {'T': 'Tensor'}.
 */
export function getIdentifierGenericMap(
    node: ts.MethodDeclaration,
    nameRemove: string): {[generic: string]: string} {
  const identifierGenericMap = {};

  node.forEachChild(child => {
    // TypeParameterDeclarations look like <T extends Tensor|NamedTensorMap>.
    if (ts.isTypeParameterDeclaration(child)) {
      let identifier;
      let generic;
      child.forEachChild(cc => {
        // Type nodes are "Tensor|NamedTensorMap"
        // Identifier nodes are "T".
        if (ts.isTypeNode(cc)) {
          generic = cc.getText();
        } else if (ts.isIdentifier(cc)) {
          identifier = cc.getText();
        }
      });
      if (identifier != null && generic != null) {
        identifierGenericMap[identifier] = generic;
      }
    }
  });

  return identifierGenericMap;
}

/**
 * Iterate over all functions in the docs.
 */
export function foreachDocFunction(
    docHeadings: DocHeading[], fn: (docFunction: DocFunction) => void) {
  docHeadings.forEach(heading => {
    heading.subheadings.forEach(subheading => {
      subheading.symbols.forEach(untypedSymbol => {
        if (untypedSymbol['isClass']) {
          const symbol = untypedSymbol as DocClass;
          symbol.methods.forEach(method => {
            fn(method);
          });
        } else {
          fn(untypedSymbol as DocFunction);
        }
      });
    });
  });
}

/**
 * Replace all types with their aliases. e.g. ShapeMap[R2] => number[]
 */
export function replaceDocTypeAliases(
    docHeadings: DocHeading[], docTypeAliases: {[type: string]: string}) {
  foreachDocFunction(docHeadings, docFunction => {
    docFunction.parameters.forEach(param => {
      param.type = replaceDocTypeAlias(param.type, docTypeAliases);
    });
    docFunction.returnType =
        replaceDocTypeAlias(docFunction.returnType, docTypeAliases);
  });
}

export function replaceDocTypeAlias(
    docTypeString: string, docTypeAliases: {[type: string]: string}): string {
  Object.keys(docTypeAliases).forEach(type => {
    if (docTypeString.indexOf(type) !== -1) {
      const re = new RegExp('\\b' + type + '\\b(\\[.+\\])?', 'g');
      docTypeString = docTypeString.replace(re, docTypeAliases[type]);
    }
  });
  return docTypeString;
}

export interface SymbolAndUrl {
  symbolName: string;
  url: string;
  type: 'function'|'class';
  namespace?: string;
}

/**
 * Adds markdown links for reference symbols in documentation, parameter types,
 * and return types. Uses @doclink aliases to link displayed symbols to another
 * symbol's documentation.
 */
export function linkSymbols(
    docs: Docs, symbols: SymbolAndUrl[], toplevelNamespace: string,
    docLinkAliases: {[symbolName: string]: string}) {
  // Find all the symbols.
  docs.headings.forEach(heading => {
    heading.subheadings.forEach(subheading => {
      subheading.symbols.forEach(symbol => {
        const namespace = toplevelNamespace + '.' +
            (symbol.namespace != null ? symbol.namespace + '.' : '');
        symbol.displayName = namespace + symbol.symbolName;

        if (symbol['isClass'] != null) {
          symbol.urlHash = `class:${symbol.displayName}`;
        } else {
          symbol.urlHash = symbol.displayName;
        }

        symbols.push({
          symbolName: symbol.symbolName,
          url: '#' + symbol.urlHash,
          type: symbol['isClass'] != null ? 'class' : 'function',
          namespace
        });
      });
    });
  });

  // Add new doc link alias symbols.
  Object.keys(docLinkAliases).forEach(docLinkAlias => {
    // Find the symbol so we can find the url hash.
    symbols.forEach(symbol => {
      if (symbol.symbolName === docLinkAliases[docLinkAlias]) {
        symbols.push({
          symbolName: docLinkAlias,
          url: symbol.url,
          type: symbol.type,
          namespace: symbol.namespace
        });
      }
    });
  });

  // Replace class documentation with links.
  docs.headings.forEach(heading => {
    heading.subheadings.forEach(subheading => {
      subheading.symbols.forEach(symbol => {
        if (symbol['isClass']) {
          symbol.documentation = replaceSymbolsWithLinks(
              symbol.documentation, symbols, true /** isMarkdown */);
        }
      });
    });
  });
  foreachDocFunction(docs.headings, method => {
    method.documentation = replaceSymbolsWithLinks(
        method.documentation, symbols, true /** isMarkdown */);
    method.returnType = replaceSymbolsWithLinks(
        method.returnType, symbols, false /** isMarkdown */);
    method.parameters.forEach(param => {
      param.documentation = replaceSymbolsWithLinks(
          param.documentation, symbols, true /** isMarkdown */);
      param.type =
          replaceSymbolsWithLinks(param.type, symbols, false /** isMarkdown */);
    });
  });
}

function replaceSymbolsWithLinks(
    input: string, symbolsAndUrls: SymbolAndUrl[],
    isMarkdown: boolean): string {
  symbolsAndUrls.forEach(symbolAndUrl => {
    const wrapper = isMarkdown ? '\`' : '\\b(?![\'\:])';
    const re = new RegExp(wrapper + symbolAndUrl.symbolName + wrapper, 'g');

    let displayText;
    if (symbolAndUrl.type === 'function') {
      displayText = symbolAndUrl.namespace ? symbolAndUrl.namespace : '';
      displayText += symbolAndUrl.symbolName + '()';
    } else {
      displayText = symbolAndUrl.symbolName;
    }

    input = input.replace(re, `[${displayText}](${symbolAndUrl.url})`);
  });
  return input;
}
