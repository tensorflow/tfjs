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
const { execFile } = require('child_process');
const { ArgumentParser } = require('argparse');
const { version } = require('./package.json');
const { resolve } = require('path');
const {
  addResultToFirestore,
  runFirestore,
  firebaseConfig,
  endFirebaseInstance
} = require('./firestore.js');
const { PromiseQueue } = require('./promise_queue');
const JSONStream = require('JSONStream');

const jsonwriter = JSONStream.stringify();
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
    console.log(`  > Running socket on: 127.0.0.1:${port}`);
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
    if (tabId === 'local') {
      continue;
    }
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
 * @param browsers The target browsers to run benchmark.
 * @param {{backend, model, numRuns, codeSnippets}} benchmarkInfo
 */
async function benchmarkAll(benchmarkInfo, browsers) {
  const allResults = [];
  for (backend of benchmarkInfo.backend) {
    for (model of benchmarkInfo.model) {
      if (model === 'codeSnippet') {
        for (codeSnippetPair of benchmarkInfo.codeSnippets) {
          console.log(
            `\nRunning codeSnippet benchmarks over ${backend} backend...`);
          const result = await benchmark({
            'benchmark': {
              'model': model,
              'numRuns': benchmarkInfo.numRuns,
              'backend': backend,
              'codeSnippet': codeSnippetPair.codeSnippet || '',
              'setupCodeSnippetEnv': codeSnippetPair.setupCodeSnippetEnv || ''
            },
            'browsers': browsers
          });
          allResults.push(result);
        }
      } else {
        console.log(
          `\nRunning ${model} model benchmarks over ${backend} backend...`);
        const result = await benchmark({
          'benchmark': {
            'model': model,
            'numRuns': benchmarkInfo.numRuns,
            'backend': backend
          },
          'browsers': browsers
        });
        allResults.push(result);
      }
    }
  }
  console.log('\nAll benchmarks complete!');
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
      `Starting benchmarks using ${cliArgs.localBuild || 'cdn'} ` +
      `dependencies...`);
  }

  const promiseQueue = new PromiseQueue(cliArgs?.maxBenchmarks ?? 9);
  const results = [];

  // Runs and gets result of each queued benchmark
  let tabIndex = 1;
  for (const tabId in config.browsers) {
    results.push(promiseQueue.add(() => {
      return runOneBenchmark(tabId, cliArgs?.maxTries, tabIndex++).then((value) => {
        value.deviceInfo = config.browsers[tabId];
        value.modelInfo = config.benchmark;
        return value;
      }).catch(error => {
        console.log(
          `${tabId} ${config.benchmark.model} ${config.benchmark.backend}`,
          error);
        return {
          error, deviceInfo: config.browsers[tabId], modelInfo: config.benchmark
        }
      });
    }));
  }

  // Optionally written to an outfile or pushed to a database once all
  // benchmarks return results
  const fulfilled = await Promise.allSettled(results);
  if (cliArgs?.outfile === 'html' || cliArgs?.outfile === 'json') {
    for (const benchmarkResult of fulfilled) {
      jsonwriter.write(benchmarkResult);
    }
  }
  if (cliArgs?.firestore) {
    await pushToFirestore(fulfilled);
  }
  console.log(
    `\n${config.benchmark?.model} model benchmark over ${config.benchmark?.backend} backend complete.\n`);
  return fulfilled;
}

function sleep(timeMs) {
  return new Promise(resolve => setTimeout(resolve, timeMs));
}

/**
 * Gets the benchmark result of a singular browser-device pairing.
 *
 * If benchmarking produces an error, the given browser-device pairing is
 * retried up to the specific max number of tries. Default is 3.
 *
 * @param tabId Indicates browser-device pairing for benchmark
 * @param triesLeft Number of tries left for a benchmark to succeed
 * @param tabIndex Indicates the sequential position for the browser-device
 *     pairing, which is used to delay initiating runner.
 * @param runOneBenchmark Function that runs a singular BrowserStack
 *     performance test
 * @param retyOneBenchmark Function that retries a singular BrowserStack
 *     performance test
 */
async function getOneBenchmarkResult(
  tabId, triesLeft, tabIndex = 0,
  runOneBenchmark = runBrowserStackBenchmark) {
  // Since karma will throw out `spawn ETXTBSY` error if initiating multiple
  // benchmark runners at the same time, adds delays between initiating runners
  // to resolve this race condition.
  const numFailed = cliArgs.maxTries - triesLeft;
  // The delay increase exponentially when benchmark fails.
  const delayInitiatingRunnerTimeMs = tabIndex * (3 ** numFailed) * 1000;
  await sleep(delayInitiatingRunnerTimeMs);

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
    const args = ['test'];
    if (tabId !== 'local') {
      args.push('--browserstack', `--browsers=${tabId}`);
    }
    if (cliArgs.localBuild) {
      args.push(`--localBuild=${cliArgs.localBuild}`)
    };
    if (cliArgs.npmVersion) {
      args.push(`--npmVersion=${cliArgs.npmVersion}`)
    };
    
    const command = `yarn ${args.join(' ')}`;
    console.log(`Running: ${command}`);
    execFile('yarn', args, { timeout: 3e5 }, (error, stdout, stderr) => {
      if (error) {
        console.log(`\n${error}`);
        console.log(`stdout: ${stdout}`);
        if (!cliArgs.cloud) {
          io.emit(
            'benchmarkComplete',
            { tabId, error: `Failed to run ${command}:\n${error}` });
        }
        return reject(`Failed to run ${command}:\n${error}`);
      }

      const errorReg = /.*\<tfjs_error\>(.*)\<\/tfjs_error\>/;
      const matchedError = stdout.match(errorReg);
      if (matchedError != null) {
        if (!cliArgs.cloud) {
          io.emit('benchmarkComplete', { tabId, error: matchedError[1] });
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
        io.emit('benchmarkComplete', { error: errorMessage })
      };
      return reject(errorMessage);
    });
  });
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
  parser.add_argument('--period', {
    help: 'runs a part of models specified in --benchmarks\'s file in a ' +
      'cycle and the part of models to run is determined by the date ' +
      'of a month. The value could be 1~31.',
    type: 'int',
    action: 'store'
  });
  parser.add_argument('--date', {
    help: 'set the date for selecting models and this works only if period ' +
      'is set. The value could be 1~31. If it is not set, the date would be ' +
      'the date at runtime).',
    type: 'int',
    action: 'store'
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
  parser.add_argument('--outfile', {
    help: 'write results to outfile. Expects \'html\' or \'json\'. ' +
      'If you set it as \'html\', benchmark_results.js will be generated ' +
      'and you could review the benchmark results by openning ' +
      'benchmark_result.html file.',
    type: 'string',
    action: 'store'
  });
  parser.add_argument('-v', '--version', { action: 'version', version });
  parser.add_argument('--localBuild', {
    help: 'local build name list, separated by comma. The name is in short ' +
      'form (in general the name without the tfjs- and backend- prefixes, ' +
      'for example webgl for tfjs-backend-webgl, core for tfjs-core). ' +
      'Example: --localBuild=webgl,core.',
    type: 'string',
    default: '',
    action: 'store'
  });
  parser.add_argument('--npmVersion', {
    help: 'specify the npm version of TFJS library to benchmark.' +
      'By default the latest version of TFJS will be benchmarked' +
      'Example: --npmVersion=4.4.0.',
    type: 'string',
    action: 'store'
  });
  cliArgs = parser.parse_args();
  console.dir(cliArgs);
}

/**
 * Get the models to benchmark for the day running the script. (All models are
 * spilted to n buckets and n === period, associated with the date of the month,
 * and the function returns a certain bucket.)
 *
 * @param models The models to schedule.
 * @param period The period to run all models.
 * @param date The value could be 1~31, and it determines the models to
 *    benchmark. By default, the date would be the date at runtime.
 */
function scheduleModels(models, period, date = new Date().getDate()) {
  if (period < 1 || period > 31) {
    throw new Error('--period must be an integer of 1~31.');
  }
  if (date <= 0 || date > 31) {
    throw new Error('--date must be an integer of 1~31.');
  }
  date = (date - 1) % period;
  const bucketSize = Math.ceil(models.length / period);
  return models.slice(date * bucketSize, (date + 1) * bucketSize);
}

/**
 * Runs a benchmark with a preconfigured file
 *
 * @param file Relative filepath to preset benchmark configuration
 * @param runBenchmark Function to run a benchmark configuration
 */
async function runBenchmarkFromFile(file, runBenchmark = benchmarkAll) {
  console.log('Running a preconfigured benchmark...');
  const { benchmark, browsers } = file;
  if (cliArgs?.period != null) {
    benchmark.model = scheduleModels(benchmark.model, cliArgs.period, cliArgs.date);
    console.log(
      `\nWill benchmark the following models: \n\t` +
      `${benchmark.model.join('\n\t')} \n`);
  } else {
    console.log(
      `\nWill benchmark all models in '${cliArgs.benchmarks}'.\n`);
  }
  await runBenchmark(benchmark, browsers);
}

async function initializeWriting() {
  if (cliArgs.firestore) {
    db = await runFirestore(firebaseConfig)
  };

  let file;
  if (cliArgs?.outfile === 'html') {
    await fs.writeFile(
      './benchmark_results.js', 'const benchmarkResults = ', 'utf8', err => {
        if (err) {
          console.log(`Error: ${err}.`);
          return reject(err);
        } else {
          return resolve();
        }
      });
    file = fs.createWriteStream('benchmark_results.js', { 'flags': 'a' });
  } else if (cliArgs?.outfile === 'json') {
    file = fs.createWriteStream('./benchmark_results.json');
  } else {
    return;
  }

  // Pipe the JSON data to the file.
  jsonwriter.pipe(file);
  console.log(`\nStart writing.`);

  // If having outfile, add a listener to Ctrl+C to finalize writing.
  process.on('SIGINT', async () => {
    await finalizeWriting();
    process.exit();
  });
}


async function finalizeWriting() {
  if (cliArgs.firestore) {
    await endFirebaseInstance();
  }

  if (cliArgs?.outfile === 'html') {
    jsonwriter.end();
    console.log('\nOutput written to ./benchmark_results.js.');
  } else if (cliArgs?.outfile === 'json') {
    jsonwriter.end();
    console.log('\nOutput written to ./benchmark_results.json.');
  }
}

/** Sets up the local or remote environment for benchmarking. */
async function prebenchmarkSetup() {
  checkBrowserStackAccount();
  await initializeWriting();

  if (!cliArgs.cloud) {
    runServer()
  };

  try {
    if (cliArgs.benchmarks) {
      const filePath = resolve(cliArgs.benchmarks);
      if (fs.existsSync(filePath)) {
        console.log(`\nFound file at ${filePath}`);
        const config = require(filePath);
        await runBenchmarkFromFile(config);
        console.log('finish')
      } else {
        throw new Error(
          `File could not be found at ${filePath}. ` +
          `Please provide a valid path.`);
      }
    }
  } finally {
    finalizeWriting();
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
exports.scheduleModels = scheduleModels;
