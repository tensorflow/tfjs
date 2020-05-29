yarn build
yarn rollup -c

cp dist/tf-backend-wasm.js ../tfjs-core/benchmarks/
cp wasm-out/tfjs-backend-wasm.wasm ../tfjs-core/benchmarks/
