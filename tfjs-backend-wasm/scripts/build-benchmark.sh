yarn build
node ./scripts/inline-worker.js
yarn rollup -c

cp dist/tf-backend-wasm.js ../e2e/benchmarks/
# cp wasm-out/tfjs-backend-wasm-threaded.js ../e2e/benchmarks/tfjs-backend-wasm.js
cp wasm-out/tfjs-backend-wasm.wasm ../e2e/benchmarks/
# cp wasm-out/tfjs-backend-wasm-simd.wasm ../e2e/benchmarks/
