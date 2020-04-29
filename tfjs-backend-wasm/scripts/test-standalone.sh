# build, then copy to benchmarks

yarn build # this creates wasm-out directory
rollup -c # this creates tf-backend-wasm.js

cp dist/tf-backend-wasm.js ../tfjs-core/benchmarks/
cp wasm-out/tfjs-backend-wasm.js ../tfjs-core/benchmarks/
cp wasm-out/tfjs-backend-wasm.worker.js ../tfjs-core/benchmarks/
cp wasm-out/tfjs-backend-wasm.wasm ../tfjs-core/benchmarks/
