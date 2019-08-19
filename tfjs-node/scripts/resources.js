/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

const fs = require('fs');
const https = require('https');
const HttpsProxyAgent = require('https-proxy-agent');
const path = require('os').platform() === 'win32' ? require('path') :
                                                    require('path').win32;
const ProgressBar = require('progress');
const tar = require('tar');
const url = require('url');
const util = require('util');
const zip = require('adm-zip');

const unlink = util.promisify(fs.unlink);

/**
 * Downloads and unpacks a given tarball or zip file at a given path.
 * @param {string} uri The path of the compressed file to download and extract.
 * @param {string} destPath The destination path for the compressed content.
 * @param {Function} callback Handler for when downloading and extraction is
 *     complete.
 */
async function downloadAndUnpackResource(uri, destPath, callback) {
  // If HTTPS_PROXY, https_proxy, HTTP_PROXY, or http_proxy is set
  const proxy = process.env['HTTPS_PROXY'] || process.env['https_proxy'] ||
      process.env['HTTP_PROXY'] || process.env['http_proxy'] || '';

  // Using object destructuring to construct the options object for the
  // http request.  the '...url.parse(targetUri)' part fills in the host,
  // path, protocol, etc from the targetUri and then we set the agent to the
  // default agent which is overridden a few lines down if there is a proxy
  const options = {...url.parse(uri), agent: https.globalAgent};

  if (proxy !== '') {
    options.agent = new HttpsProxyAgent(proxy);
  }

  const request = https.get(options, response => {
    const bar = new ProgressBar('[:bar] :rate/bps :percent :etas', {
      complete: '=',
      incomplete: ' ',
      width: 30,
      total: parseInt(response.headers['content-length'], 10)
    });

    if (uri.endsWith('.zip')) {
      // Save zip file to disk, extract, and delete the downloaded zip file.
      const tempFileName = path.join(__dirname, '_tmp.zip');
      const outputFile = fs.createWriteStream(tempFileName);

      response.on('data', chunk => bar.tick(chunk.length))
          .pipe(outputFile)
          .on('close', async () => {
            const zipFile = new zip(tempFileName);
            zipFile.extractAllTo(destPath, true /* overwrite */);

            await unlink(tempFileName);

            if (callback !== undefined) {
              callback();
            }
          });
    } else if (uri.endsWith('.tar.gz')) {
      response.on('data', chunk => bar.tick(chunk.length))
          .pipe(tar.x({C: destPath, strict: true}))
          .on('close', () => {
            if (callback !== undefined) {
              callback();
            }
          });
    } else {
      throw new Error(`Unsupported packed resource: ${uri}`);
    }
  });
  request.end();
}

module.exports = {downloadAndUnpackResource};
