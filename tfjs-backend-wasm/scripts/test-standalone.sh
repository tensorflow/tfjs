# build, then copy to benchmarks

yarn build # this creates wasm-out directory

node ./scripts/inline-worker.js

rollup -c # this creates tf-backend-wasm.js

cp dist/tf-backend-wasm.js ../tfjs-core/benchmarks/
cp wasm-out/tfjs-backend-wasm-threaded.js ../tfjs-core/benchmarks/tfjs-backend-wasm.js
cp wasm-out/tfjs-backend-wasm-threaded.wasm ../tfjs-core/benchmarks/
