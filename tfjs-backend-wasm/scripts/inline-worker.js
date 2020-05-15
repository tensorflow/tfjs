const fs = require('fs');

const workerContents = fs.readFileSync('./wasm-out/tfjs-backend-wasm.worker.js', "utf8");

const fileContents = `export const wasmWorkerContents = '${workerContents.trim()}';`;

fs.writeFile('./wasm-out/tfjs-backend-wasm.worker.ts', fileContents, function(err) {
  console.log("dobne");
});
