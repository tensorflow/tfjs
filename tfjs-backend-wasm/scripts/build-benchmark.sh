yarn build
yarn rollup -c

cp dist/tf-backend-wasm.js ../e2e/benchmarks/local-benchmark/
cp wasm-out/tfjs-backend-wasm.wasm ../e2e/benchmarks/local-benchmark/
cp wasm-out/tfjs-backend-wasm-simd.wasm ../e2e/benchmarks/local-benchmark/
cp wasm-out/tfjs-backend-wasm-threaded-simd.wasm ../e2e/benchmarks/local-benchmark/
