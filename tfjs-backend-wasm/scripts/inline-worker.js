const fs = require('fs');

const workerContents = fs.readFileSync('./wasm-out/tfjs-backend-wasm-threaded.worker.js', "utf8");

const fileContents = `export const wasmWorkerContents = '${workerContents.trim()}';`;

fs.writeFile('./wasm-out/tfjs-backend-wasm.worker.ts', fileContents, function(err) {
  console.log("dobne");
});


// const moduleContents = fs.readFileSync('./wasm-out/tfjs-backend-wasm-threaded.js', 'utf-8');

// // const moduleFileContents = `export const wasmModuleContents = '${moduleContents.trim().replace(/\r?\n|\r/g, " ")}';`;

// const moduleFileContents = 'export const wasmModuleContents = `' + moduleContents.trim().replace(/\r?\n|\r/g, " ") + '`;';

// fs.writeFile('./wasm-out/tfjs-backend-wasm.ts', moduleFileContents, function(err) {
//   console.log("done done");
// });
