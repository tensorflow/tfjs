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

const io = socketio(app);

app.listen(port, () => {
  console.log(`  > Running socket on port: ${port}`);
});

io.on('connection', socket => {
  socket.on('run', benchmark);
});

function benchmark(config) {
  // TODO:
  // 1. Write browsers.json.
  // 2. Write benchmark parameter config.
  fs.writeFileSync(
      './benchmark_parameters.json', JSON.stringify(config.benchmark, null, 2));

  console.log(`Start benchmarking.`);
  exec('yarn test', (error, stdout, stderr) => {
    if (error) {
      console.log(error);
      io.emit('benchmarkComplete', {error: 'Failed to run yarn test.'});
      return;
    }

    const errorReg = /.*\<tfjs_error\>(.*)\<\/tfjs_error\>/;
    const matchedError = stdout.match(errorReg);
    if (matchedError != null) {
      io.emit('benchmarkComplete', {error: matchedError[1]});
      return;
    }

    const resultReg = /.*\<tfjs_benchmark\>(.*)\<\/tfjs_benchmark\>/;
    const matchedResult = stdout.match(resultReg);
    if (matchedResult != null) {
      const benchmarkResult = JSON.parse(matchedResult[1]);
      io.emit('benchmarkComplete', benchmarkResult);
      return;
    }

    io.emit('benchmarkComplete', {
      error: 'Did not find benchmark results from the logs ' +
          'of the benchmark test (benchmark_models.js).'
    });
  });
}
