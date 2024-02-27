// Copyright 2022 Google LLC. All Rights Reserved.
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

const express = require('express');
const {ArgumentParser, ArgumentDefaultsHelpFormatter} = require('argparse');
const JSZip = require('jszip');
const fetch = require('node-fetch');
const path = require('node:path');

const DEFAULT_VERSION = '20221122-100434';
const TFJS_DEBUGGER_BUNDLE_URL_BASE =
    'https://storage.googleapis.com/tfweb/tfjs-debugger-bundle';
const EXTS_WITH_ARRAY_BUFFER_CONTENT = ['.ttf'];

const app = express();
const debuggerStaticFile = {};

function main(args) {
  const port = parseInt(args['port']);
  const version = args['version'];

  app.get('/*', (req, res) => {
    // Remove leading '/'.
    let filePath = req.path.substring(1);
    if (filePath === '') {
      filePath = 'index.html';
    }

    // Send file from the unzipped tfjs-debugger bundle from memory.
    if (debuggerStaticFile[filePath]) {
      res.contentType(path.basename(filePath));
      res.send(debuggerStaticFile[filePath]);
    }
    // Send other files from the local file system.
    else {
      const tfjsRoot = __dirname.replace('/scripts', '/');
      res.sendFile(tfjsRoot + filePath);
    }
  });

  app.listen(port, async () => {
    // On server start-up, fetch the zipped debugger bundle and unzip in memory.
    console.log('Fetching tfjs debugger static files...');
    await fetchAndUnzipTfjsDebuggerBundle(version);
    console.log('Done');
    console.log(`Local debugger server started at http://localhost:${
        port}/?bv__0=Local%20build&bv__1=`);
  });
}

async function fetchAndUnzipTfjsDebuggerBundle(version) {
  let resp;
  try {
    resp = await new Promise(resolve => {
      const fileUrl =
          `${TFJS_DEBUGGER_BUNDLE_URL_BASE}/tfjs-debugger_${version}.zip`;
      fetch(fileUrl).then(response => {
        if (!response.ok) {
          throw new Error(`Failed to load bundle: ${fileUrl}`);
        }
        resolve(response);
      })
    });
  } catch (e) {
    console.error(e.message);
  }
  const buffer = await resp.buffer();

  // Read zip objects.
  const zipObjects = await new Promise(resolve => {
    const zipObjects = [];
    JSZip.loadAsync(buffer).then((zip) => {
      zip.folder('dist').forEach((relativePath, zipObject) => {
        zipObjects.push(zipObject);
      });
      resolve(zipObjects);
    })
  });

  // Read and index files content.
  for (const zipObject of zipObjects) {
    await new Promise(resolve => {
      const name = zipObject.name;
      zipObject
          .async(
              EXTS_WITH_ARRAY_BUFFER_CONTENT.some(ext => name.endsWith(ext)) ?
                  'nodebuffer' :
                  'text')
          .then(content => {
            const fileName = name.replace('dist/', '');
            debuggerStaticFile[fileName] = content;
            resolve();
          });
    });
  }
}

const parser = new ArgumentParser({
  description: 'Run tfjs-debugger locally that supports loading local packages',
});

parser.addArgument('--port', {
  help: 'Server port',
  defaultValue: 9876,
  type: 'int',
});

parser.addArgument('--version', {
  help: `The version of the bundle. Default: ${DEFAULT_VERSION}`,
  defaultValue: DEFAULT_VERSION,
  type: 'string',
});

main(parser.parseArgs());
