/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import * as ts from 'typescript';

process.on('unhandledRejection', ex => {
  throw ex;
});

// Used for logging the number of snippets that have been found.
let snippetCount = 0;
// Used for counting the number of errors that have been found.
let errorCount = 0;

/**
 * Parse and evaluate snippets for the src/index.ts from where this script is
 * run.
 * @param tf The TensorFlow.js module to use when evaluating snippets. If used
 *     outside core, this should be a union of core and the separate package.
 *     This is unused here but is used in eval() of the snippets.
 */
// tslint:disable-next-line:no-any
export async function parseAndEvaluateSnippets(tf: any) {
  const index = path.join(process.cwd(), 'src/index.ts');
  const tsconfigPath = path.join(process.cwd(), 'tsconfig.json');

  // Use the same compiler options that we use to compile the library
  // here.
  const tsconfig = JSON.parse(fs.readFileSync(tsconfigPath, 'utf8'));

  delete tsconfig.compilerOptions.moduleResolution;
  const program = ts.createProgram([index], tsconfig.compilerOptions);

  const checker = program.getTypeChecker();

  for (const sourceFile of program.getSourceFiles()) {
    if (!sourceFile.isDeclarationFile) {
      const children = sourceFile.getChildren();
      for (let i = 0; i < children.length; i++) {
        await visit(tf, checker, children[i], sourceFile);
      }
    }
  }

  if (errorCount === 0) {
    console.log(`Parsed and evaluated ${snippetCount} snippets successfully.`);
  } else {
    console.log(
        `Evaluated ${snippetCount} snippets with ${errorCount} errors.`);
    process.exit(1);
  }
}

async function visit(
    // tslint:disable-next-line:no-any
    tf: any, checker: ts.TypeChecker, node: ts.Node,
    sourceFile: ts.SourceFile) {
  const children = node.getChildren();
  for (let i = 0; i < children.length; i++) {
    await visit(tf, checker, children[i], sourceFile);
  }

  if (ts.isClassDeclaration(node) || ts.isFunctionDeclaration(node) ||
      ts.isMethodDeclaration(node) || ts.isInterfaceDeclaration(node)) {
    const symbol = checker.getSymbolAtLocation(node.name);
    const jsdoc = getJSDocTag(symbol);
    if (jsdoc == null) {
      return;
    }
    // Ignore snippets of methods that have been marked with ignoreCI.
    if (jsdoc['ignoreCI']) {
      return;
    }

    const documentation = symbol.getDocumentationComment(checker);
    if (documentation == null) {
      return;
    }
    for (let i = 0; i < documentation.length; i++) {
      const doc = documentation[i];
      const re = /```js.*?```/gs;
      const matches = re.exec(doc.text);
      if (matches == null) {
        return;
      }

      for (let k = 0; k < matches.length; k++) {
        snippetCount++;

        const match = matches[k];
        const lines = match.split('\n');
        const evalLines: string[] = [];
        for (let j = 0; j < lines.length; j++) {
          let line = lines[j];
          if (line.startsWith('```js')) {
            line = line.substring('```js'.length);
          }
          if (line.endsWith('```')) {
            line = line.substring(0, line.length - '```'.length);
          }
          line = line.trim();
          if (line.startsWith('*')) {
            line = line.substring(1).trim();
          }
          evalLines.push(line);
        }

        const srcCode = evalLines.join('\n');

        const evalString = '(async function runner() { try { ' + srcCode +
            '} catch (e) { reportError(e); } })()';

        const oldLog = console.log;
        const oldWarn = console.warn;

        const reportError = (e: string|Error) => {
          oldLog();
          oldLog(`Error executing snippet for ${symbol.name} at ${
              sourceFile.fileName}`);
          oldLog();
          oldLog(`\`\`\`js${srcCode}\`\`\``);
          oldLog();

          console.error(e);
          errorCount++;
        };

        // Overrwrite console.log so we don't spam the console.
        console.log = (msg: string) => {};
        console.warn = (msg: string) => {};
        try {
          await eval(evalString);
        } catch (e) {
          reportError(e);
        }
        console.log = oldLog;
        console.warn = oldWarn;
      }
    }
  }
}

interface JSDoc {
  namespace?: string;
  ignoreCI?: boolean;
}

function getJSDocTag(symbol: ts.Symbol): JSDoc {
  const tags = symbol.getJsDocTags();
  for (let i = 0; i < tags.length; i++) {
    const jsdocTag = tags[i];
    if (jsdocTag.name === 'doc' && jsdocTag.text != null) {
      const json = convertDocStringToDocInfoObject(jsdocTag.text.trim());
      return json;
    }
  }
  return null;
}

function convertDocStringToDocInfoObject(docString: string): JSDoc {
  const jsonString =
      docString.replace(/([a-zA-Z0-9]+):/g, '"$1":').replace(/\'/g, '"');
  return JSON.parse(jsonString);
}
