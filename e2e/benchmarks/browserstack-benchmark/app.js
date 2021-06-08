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
const {execFile} = require('child_process');
const { ArgumentParser } = require('argparse');
const { version } = require('./package.json');

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
    const availableBrowsers = require('./browser_list.json');
    socket.emit('availableBrowsers', availableBrowsers);
    socket.on('run', benchmark);
  });
}

/**
 * Supplement the browser configurations and create `browsers.json` and
 * `benchmark_parameters.json` configuration files for karma.
 *
 * @param {{browsers, benchmark}} config
 */
function setupBenchmarkEnv(config) {
  // Write the map (tabId - browser setting) to `./browsers.json`.
  for (const tabId in config.browsers) {
    const browser = config.browsers[tabId];
    browser.base = 'BrowserStack';
    // For mobile devices, we would use real devices instead of emulators.
    if (browser.os === 'ios' || browser.os === 'android') {
      browser.real_mobile = true;
    }
  }
  fs.writeFileSync('./browsers.json', JSON.stringify(config.browsers, null, 2));

  // Write benchmark parameters to './benchmark_parameters.json'.
  fs.writeFileSync(
      './benchmark_parameters.json', JSON.stringify(config.benchmark, null, 2));
}

/**
 * Run model benchmark on BrowserStack.
 *
 * The benchmark configuration object contains two objects:
 * - `browsers`: Each key-value pair represents a browser instance to be
 * benchmarked. The key is a unique string id/tabId (assigned by the webpage)
 * for the browser instance, while the value is the browser configuration.
 *
 * - `benchmark`: An object with the following properties:
 *  - `model`: The name of model (registed at
 * 'tfjs/e2e/benchmarks/model_config.js') or `custom`.
 *  - modelUrl: The URL to the model description file. Only applicable when the
 * `model` is `custom`.
 *  - `numRuns`: The number of rounds for model inference.
 *  - `backend`: The backend to be benchmarked on.
 *
 *
 * @param {{browsers, benchmark}} config Benchmark configuration.
 */
function benchmark(config) {
  console.log('Preparing configuration files for the test runner.');
  setupBenchmarkEnv(config);

  console.log(`Start benchmarking.`);
  for (const tabId in config.browsers) {
    const args = ['test', '--browserstack', `--browsers=${tabId}`];
    const command = `yarn ${args.join(' ')}`;
    console.log(`Running: ${command}`);

    execFile('yarn', args, (error, stdout, stderr) => {
      console.log(`benchmark ${tabId} completed.`);
      if (error) {
        console.log(error);
        io.emit(
            'benchmarkComplete',
            {tabId, error: `Failed to run ${command}:\n${error}`});
        return;
      }

      const errorReg = /.*\<tfjs_error\>(.*)\<\/tfjs_error\>/;
      const matchedError = stdout.match(errorReg);
      if (matchedError != null) {
        io.emit('benchmarkComplete', {tabId, error: matchedError[1]});
        return;
      }

      const resultReg = /.*\<tfjs_benchmark\>(.*)\<\/tfjs_benchmark\>/;
      const matchedResult = stdout.match(resultReg);
      if (matchedResult != null) {
        const benchmarkResult = JSON.parse(matchedResult[1]);
        benchmarkResult.tabId = tabId;
        io.emit('benchmarkComplete', benchmarkResult);
        return;
      }

      io.emit('benchmarkComplete', {
        error: 'Did not find benchmark results from the logs ' +
            'of the benchmark test (benchmark_models.js).'
      });
    });
  }
}

/** Set up --help menu to show available optional commands */
function setUpHelpMessage() {
  const parser = new ArgumentParser({
    description: 'The following commands are available:'
  });

  parser.add_argument('-v', '--version', { action: 'version', version });
  console.dir(parser.parse_args());
}

setUpHelpMessage();
checkBrowserStackAccount();
runServer();
