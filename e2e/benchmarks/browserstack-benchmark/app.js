const http = require('http');
const socketio = require('socket.io');
const fs = require('fs');
const {exec} = require('child_process');

const port = process.env.PORT || 8001;

const app = http.createServer((request, response) => {
  const url = request.url === '/' ? '/index.html' : request.url;
  fs.readFile(__dirname + url, (err, data) => {
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
  console.log(`Start benchmarking.`);
  exec('yarn test', (err, stdout, stderr) => {
    if (err) {
      console.log(err);
      return;
    }
    const re = /.*\<benchmark\>(.*)\<\/benchmark\>/;
    const benchmarkResultStr = stdout.match(re)[1];
    const benchmarkResult = JSON.parse(benchmarkResultStr);
    io.emit('benchmarkComplete', benchmarkResult);
  });
}
