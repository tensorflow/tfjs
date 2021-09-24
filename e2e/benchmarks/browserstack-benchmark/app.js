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
const {ArgumentParser} = require('argparse');
const {version} = require('./package.json');
const {resolve} = require('path');
const {
  addResultToFirestore,
  runFirestore,
  firebaseConfig,
  endFirebaseInstance
} = require('./firestore.js');

const port = process.env.PORT || 8001;
let io;
let parser;
let cliArgs;
let db;

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
 * Creates and runs benchmark configurations for each model-backend pairing.
 *
 * @param {{browsers, benchmark}} config
 */
async function benchmarkAll(config) {
  const allResults = [];
  const benchmarkInfo = config.benchmark;

  for (backend of benchmarkInfo.backend) {
    for (model of benchmarkInfo.model) {
      console.log(
          `\nRunning ${model} model benchmarks over ${backend} backend...`);
      const result = await benchmark({
        'benchmark': {
          'model': model,
          'numRuns': benchmarkInfo.numRuns,
          'backend': backend
        },
        'browsers': config.browsers
      });
      allResults.push(result);
    }
  }
  console.log('\nAll benchmarks complete!');
  endFirebaseInstance();
  return allResults;
}

/**
 * Run model benchmark on BrowserStack.
 *
 * Each browser-device pairing is benchmarked in parallel. Results are sent to
 * the webpage staggered as they are returned to the server.
 *
 * The benchmark configuration object contains two objects:
 * - `browsers`: Each key-value pair represents a browser instance to be
 * benchmarked. The key is a unique string id/tabId (assigned by the webpage)
 * for the browser instance, while the value is the browser configuration.
 *
 * - `benchmark`: An object with the following properties:
 *  - `model`: The name of model (registed at
 * 'tfjs/e2e/benchmarks/model_config.js') or `custom`.
 *  - modelUrl: The URL to the model description file. Only applicable when
 * the `model` is `custom`.
 *  - `numRuns`: The number of rounds for model inference.
 *  - `backend`: The backend to be benchmarked on.
 *
 *
 * @param {{browsers, benchmark}} config Benchmark configuration
 * @param runOneBenchmark Function that benchmarks one browser-device pair
 */
async function benchmark(config, runOneBenchmark = getOneBenchmarkResult) {
  console.log('Preparing configuration files for the test runner.\n');
  setupBenchmarkEnv(config);
  if (require.main === module) {
    console.log(
        `Starting benchmarks using ${cliArgs?.webDeps ? 'cdn' : 'local'} ` +
        `dependencies...`);
  }

  const results = [];
  let numActiveBenchmarks = 0;
  // Runs and gets result of each queued benchmark
  for (const tabId in config.browsers) {
    numActiveBenchmarks++;
    results.push(runOneBenchmark(tabId, cliArgs?.maxTries).then((value) => {
      value.deviceInfo = config.browsers[tabId];
      value.modelInfo = config.benchmark;
      return value;
    }));

    // Waits for specified # of benchmarks to complete before running more
    if (cliArgs?.maxBenchmarks && numActiveBenchmarks >= cliArgs.maxBenchmarks) {
      numActiveBenchmarks = 0;
      await Promise.allSettled(results);
    }
  }

  // Optionally written to an outfile or pushed to a database once all
  // benchmarks return results
  const fulfilled = await Promise.allSettled(results);
  if (cliArgs?.outfile) {
    await write('./benchmark_results.json', fulfilled);
  } else {
    console.log('\Benchmarks complete.\n');
  }
  if (cliArgs?.firestore) {
    await pushToFirestore(fulfilled);
  }
  return fulfilled;
}

/**
 * Gets the benchmark result of a singular browser-device pairing.
 *
 * If benchmarking produces an error, the given browser-device pairing is
 * retried up to the specific max number of tries. Default is 3.
 *
 * @param tabId Indicates browser-device pairing for benchmark
 * @param triesLeft Number of tries left for a benchmark to succeed
 * @param runOneBenchmark Function that runs a singular BrowserStack
 *     performance test
 * @param retyOneBenchmark Function that retries a singular BrowserStack
 *     performance test
 */
async function getOneBenchmarkResult(
    tabId, triesLeft, runOneBenchmark = runBrowserStackBenchmark) {
  triesLeft--;
  try {
    const result = await runOneBenchmark(tabId);
    console.log(`${tabId} benchmark succeeded.`);
    return result;
  } catch (err) {
    // Retries benchmark until resolved or until no retries left
    if (triesLeft > 0) {
      console.log(`Retrying ${tabId} benchmark. ${triesLeft} tries left...`);
      return await getOneBenchmarkResult(tabId, triesLeft, runOneBenchmark);
    } else {
      console.log(`${tabId} benchmark failed.`);
      throw err;
    }
  }
}

/**
 * Run benchmark for singular browser-device combination.
 *
 * This function utilizes a promise that is fulfilled once the corresponding
 * result is returned from BrowserStack.
 *
 * @param tabId Indicates browser-device pairing for benchmark
 */
function runBrowserStackBenchmark(tabId) {
  return new Promise((resolve, reject) => {
    const args = ['test', '--browserstack', `--browsers=${tabId}`];
    if (cliArgs.webDeps) {
      args.push('--cdn')
    };
    const command = `yarn ${args.join(' ')}`;
    console.log(`Running: ${command}`);

    execFile('yarn', args, (error, stdout, stderr) => {
      if (error) {
        console.log(`\n${error}`);
        console.log(`stdout: ${stdout}`);
        if (!cliArgs.cloud) {
          io.emit(
              'benchmarkComplete',
              {tabId, error: `Failed to run ${command}:\n${error}`});
        }
        return reject(`Failed to run ${command}:\n${error}`);
      }

      const errorReg = /.*\<tfjs_error\>(.*)\<\/tfjs_error\>/;
      const matchedError = stdout.match(errorReg);
      if (matchedError != null) {
        if (!cliArgs.cloud) {
          io.emit('benchmarkComplete', {tabId, error: matchedError[1]});
        }
        return reject(matchedError[1]);
      }

      const resultReg = /.*\<tfjs_benchmark\>(.*)\<\/tfjs_benchmark\>/;
      const matchedResult = stdout.match(resultReg);
      if (matchedResult != null) {
        const benchmarkResult = JSON.parse(matchedResult[1]);
        benchmarkResult.tabId = tabId;
        if (!cliArgs.cloud) {
          io.emit('benchmarkComplete', benchmarkResult)
        };
        return resolve(benchmarkResult);
      }

      const errorMessage = 'Did not find benchmark results from the logs ' +
          'of the benchmark test (benchmark_models.js).';
      if (!cliArgs.cloud) {
        io.emit('benchmarkComplete', {error: errorMessage})
      };
      return reject(errorMessage);
    });
  });
}

/**
 * Writes a passed message to a passed JSON file.
 *
 * @param filePath Relative filepath of target file
 * @param msg Message to be written
 */
function write(filePath, msg) {
  return new Promise((resolve, reject) => {
    fs.writeFile(filePath, JSON.stringify(msg, null, 2), 'utf8', err => {
      if (err) {
        console.log(`Error: ${err}.`);
        return reject(err);
      } else {
        console.log('\nOutput written.');
        return resolve();
      }
    });
  })
}

/**
 * Pushes all benchmark results to Firestore.
 *
 * @param benchmarkResults List of all benchmark results
 */
async function pushToFirestore(benchmarkResults) {
  let firestoreResults = [];
  let numRejectedPromises = 0;
  console.log('\Pushing results to Firestore...');
  for (result of benchmarkResults) {
    if (result.status == 'fulfilled') {
      firestoreResults.push(
          addResultToFirestore(db, result.value.tabId, result.value));
    } else if (result.status == 'rejected') {
      numRejectedPromises++;
    }
  }
  return await Promise.allSettled(firestoreResults).then(() => {
    console.log(
        `Encountered ${numRejectedPromises} rejected promises that were not ` +
        `added to the database.`);
  });
}

/** Set up --help menu for file description and available optional commands */
function setupHelpMessage() {
  parser = new ArgumentParser({
    description: 'This file launches a server to connect to BrowserStack ' +
        'so that the performance of a TensorFlow model on one or more ' +
        'browsers can be benchmarked.'
  });
  parser.add_argument('--benchmarks', {
    help: 'run a preconfigured benchmark from a user-specified JSON',
    action: 'store'
  });
  parser.add_argument('--cloud', {
    help: 'runs GCP compatible version of benchmarking system',
    action: 'store_true'
  });
  parser.add_argument('--maxBenchmarks', {
    help: 'the maximum number of benchmarks run in parallel',
    type: 'int',
    default: 5,
    action: 'store'
  });
  parser.add_argument('--maxTries', {
    help: 'the maximum number of times a given benchmark is tried befor it ' +
        'officially fails',
    type: 'int',
    default: 3,
    action: 'store'
  });
  parser.add_argument('--firestore', {
    help: 'Store benchmark results in Firestore database',
    action: 'store_true'
  });
  parser.add_argument(
      '--outfile', {help: 'write results to outfile', action: 'store_true'});
  parser.add_argument('-v', '--version', {action: 'version', version});
  parser.add_argument('--webDeps', {
    help: 'utilizes public, web hosted dependencies instead of local versions',
    action: 'store_true'
  });
  cliArgs = parser.parse_args();
  console.dir(cliArgs);
}

/**
 * Runs a benchmark with a preconfigured file
 *
 * @param file Relative filepath to preset benchmark configuration
 * @param runBenchmark Function to run a benchmark configuration
 */
function runBenchmarkFromFile(file, runBenchmark = benchmarkAll) {
  console.log('Running a preconfigured benchmark...');
  runBenchmark(file);
}

/** Sets up the local or remote environment for benchmarking. */
async function prebenchmarkSetup() {
  checkBrowserStackAccount();
  if (cliArgs.firestore) {
    db = await runFirestore(firebaseConfig)
  };
  if (!cliArgs.cloud) {
    runServer()
  };
  if (cliArgs.benchmarks) {
    const filePath = resolve(cliArgs.benchmarks);
    if (fs.existsSync(filePath)) {
      console.log(`\nFound file at ${filePath}`);
      const config = require(filePath);
      runBenchmarkFromFile(config);
    } else {
      throw new Error(
          `File could not be found at ${filePath}. ` +
          `Please provide a valid path.`);
    }
  }
}

/* Only run this code if app.js is called from the command line */
if (require.main === module) {
  setupHelpMessage();
  prebenchmarkSetup();
}

exports.runBenchmarkFromFile = runBenchmarkFromFile;
exports.getOneBenchmarkResult = getOneBenchmarkResult;
exports.benchmark = benchmark;
exports.write = write;
