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
import * as HandleBars from 'handlebars';
import * as hljs from 'highlight.js';
import * as MarkdownIt from 'markdown-it';
import * as minimist from 'minimist';
import * as mkdirp from 'mkdirp';
import * as shell from 'shelljs';
import * as ts from 'typescript';

import * as parser from './api-parser';
import * as util from './api-util';

const argv = minimist(process.argv.slice(2));

const TOPLEVEL_NAMESPACE = 'dl';
const API_TEMPLATE_PATH = './website/api/index.html';
const HTML_OUT_DIR = argv.o || '/tmp/deeplearn-new-website/api/';

console.log('Building API docs to: ' + HTML_OUT_DIR);

shell.mkdir('-p', HTML_OUT_DIR);

let bundleJsPath;
if (argv['master']) {
  // When using --master, build a deeplearn.js bundle if it doesn't exist. This
  // is mainly for development.
  if (!fs.existsSync(HTML_OUT_DIR + 'deeplearn.js')) {
    shell.exec('./scripts/build-standalone.sh');
    shell.cp('./dist/deeplearn.js', HTML_OUT_DIR);
  }
  bundleJsPath = 'deeplearn.js';
} else {
  // Read version and point to it.
  const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  bundleJsPath = `https://cdn.jsdelivr.net/npm/deeplearn@${pkg.version}`;
}
console.log(`Using bundle path ${bundleJsPath}.`);

const {docs, docLinkAliases} = parser.parse();
docs.bundleJsPath = bundleJsPath;

// Predefine some custom type links.
const symbols: util.SymbolAndUrl[] = [
  {
    symbolName: 'TypedArray',
    url: 'https://developer.mozilla.org/en-US/docs/Web/' +
        'JavaScript/Reference/Global_Objects/TypedArray',
    type: 'class'
  },
  {
    symbolName: 'ImageData',
    url: 'https://developer.mozilla.org/en-US/docs/Web/API/ImageData',
    type: 'class'
  },
  {
    symbolName: 'HTMLImageElement',
    url: 'https://developer.mozilla.org/en-US/docs/Web/API/HTMLImageElement',
    type: 'class'
  },
  {
    symbolName: 'HTMLCanvasElement',
    url: 'https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement',
    type: 'class'
  },
  {
    symbolName: 'HTMLVideoElement',
    url: 'https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement',
    type: 'class'
  }
];
util.linkSymbols(docs, symbols, TOPLEVEL_NAMESPACE, docLinkAliases);

const md = new MarkdownIt({
  highlight(str, lang) {
    if (lang === 'js' && hljs.getLanguage(lang)) {
      const highlighted = hljs.highlight(lang, str).value;
      return '<pre class="hljs"><code class="hljs language-js">' + highlighted +
          '</code></pre>\n';
    }

    return '';  // use external default escaping
  }
});

// Add some helper functions
HandleBars.registerHelper('markdown', attr => {
  if (attr) {
    return md.render(attr);
  }
});

// Renders a string to markdown but removes the outer <p> tag
HandleBars.registerHelper('markdownInner', attr => {
  if (attr) {
    const asMd =
        md.render(attr.trim()).replace(/<p>/, '').replace(/(<\/p>\s*)$/, '');

    return asMd;
  }
});

// Write the HTML.
const htmlFilePath = HTML_OUT_DIR + 'index.html';
const template = fs.readFileSync(API_TEMPLATE_PATH, 'utf8');

const html = HandleBars.compile(template)(docs);
fs.writeFileSync(htmlFilePath, html);

// Compute some statics and render them.
const {headingsCount, subheadingsCount, methodCount} =
    util.computeStatistics(docs);
console.log(
    `API reference written to ${htmlFilePath}\n` +
    `Found: \n` +
    `  ${headingsCount} headings\n` +
    `  ${subheadingsCount} subheadings\n` +
    `  ${methodCount} methods`);
