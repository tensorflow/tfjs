/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

const http = require('http');
const socketio = require('socket.io');
const fs = require('fs');
const path = require('path');
const {exec} = require('child_process');

const port = process.env.PORT || 8001;
let io;

function checkBrowserStackAccount() {
  if (process.env.BROWSERSTACK_USERNAME == null ||
      process.env.BROWSERSTACK_ACCESS_KEY == null) {
    throw new Error(
        `Please export your BrowserStack username and access key by running` +
        `the following commands in the terminal:
          export BROWSERSTACK_USERNAME=YOUR_USERNAME
          export BROWSERSTACK_ACCESS_KEY=YOUR_ACCESS_KEY`);
  }
}

function runServer() {
  const app = http.createServer((request, response) => {
    const url = request.url === '/' ? '/index.html' : request.url;
    let filePath = path.join(__dirname, url);
    if (!fs.existsSync(filePath)) {
      filePath = path.join(__dirname, '../', url);
    }
    fs.readFile(filePath, (err, data) => {
      if (err) {
        response.writeHead(404);
        response.end(JSON.stringify(err));
        return;
      }
      response.writeHead(200);
      response.end(data);
    });
  });
  app.listen(port, () => {
    console.log(`  > Running socket on port: ${port}`);
  });

  io = socketio(app);
  io.on('connection', socket => {
    socket.on('run', benchmark);
  });
}

function benchmark(config) {
  const benchmarkResult = require('./benchmark_res.json');
  benchmarkResult.tabName = config.tabName;
  io.emit('benchmarkComplete', benchmarkResult);

  // console.log('Preparing configuration files for the test runner.');
  // // TODO:
  // // 1. Write browsers.json.
  // // Write the browsers to benchmark to `./browsers.json`.
  // const browser = config.browser;
  // browser.base = 'BrowserStack';
  // // For mobile devices, we would use real devices instead of emulators.
  // if (browser.os === 'ios' || browser.os === 'android') {
  //   browser.real_mobile = true;
  // }
  // fs.writeFileSync('./browsers.json', JSON.stringify([browser], null, 2));

  // // 2. Write benchmark parameter config.
  // fs.writeFileSync(
  //     './benchmark_parameters.json', JSON.stringify(config.benchmark, null,
  //     2));

  // console.log(`Start benchmarking.`);
  // exec('yarn test --browserstack', (error, stdout, stderr) => {
  //   console.log(`benchmark completed.`);
  //   if (error) {
  //     console.log(error);
  //     io.emit(
  //         'benchmarkComplete',
  //         {error: `Failed to run 'yarn test --browserstack':\n\n${error}`});
  //     return;
  //   }

  //   const errorReg = /.*\<tfjs_error\>(.*)\<\/tfjs_error\>/;
  //   const matchedError = stdout.match(errorReg);
  //   if (matchedError != null) {
  //     io.emit('benchmarkComplete', {error: matchedError[1]});
  //     return;
  //   }

  //   const resultReg = /.*\<tfjs_benchmark\>(.*)\<\/tfjs_benchmark\>/;
  //   const matchedResult = stdout.match(resultReg);
  //   if (matchedResult != null) {
  //     const benchmarkResult = JSON.parse(matchedResult[1]);
  //     benchmarkResult.tabName = config.tabName;
  //     io.emit('benchmarkComplete', benchmarkResult);
  //     return;
  //   }

  //   io.emit('benchmarkComplete', {
  //     error: 'Did not find benchmark results from the logs ' +
  //         'of the benchmark test (benchmark_models.js).'
  //   });
  // });
}

checkBrowserStackAccount();
runServer();
