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
import * as MarkdownIt from 'markdown-it';
import * as minimist from 'minimist';
import * as mkdirp from 'mkdirp';
import * as shell from 'shelljs';
import * as ts from 'typescript';

import * as parser from './api-parser';
import * as util from './api-util';

const TOPLEVEL_NAMESPACE = 'dl';
const API_TEMPLATE_PATH = './website/api/index.html';
const HTML_OUT_DIR = '/tmp/deeplearn-new-website/api/';

shell.mkdir('-p', HTML_OUT_DIR);

const {docs, docLinkAliases} = parser.parse();

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

const md = new MarkdownIt();

// Add some helper functions
HandleBars.registerHelper('markdown', attr => {
  return md.render(attr);
});

// Renders a string to markdown but removes the outer <p> tag
HandleBars.registerHelper('markdownInner', attr => {
  const asMd =
      md.render(attr.trim()).replace(/<p>/, '').replace(/(<\/p>\s*)$/, '');

  return asMd;
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
