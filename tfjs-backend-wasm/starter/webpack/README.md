# Starter project for WebPack

A minimal starter project using WebPack and the WASM backend.
See [Using Bundlers](../../README.md#using-bundlers) section in the main readme
for details on how we serve the WASM backend on NPM.

To serve the `.wasm` file, we use the `file-loader` plugin and added the
following to `webpack.config.js`, which tells WebPack to provide the serving url
when importing `.wasm` files:

```js
module: {
  rules: [
    {
      test: /\.wasm$/i,
      type: 'javascript/auto',
      use: [
        {
          loader: 'file-loader',
        },
      ],
    },
  ],
}
```

Then we obtain the final serving path of the WASM binaries that were shipped on
NPM, and use `setWasmPaths` to let the library know the serving locations:

```ts
import {setWasmPaths} from '@tensorflow/tfjs-backend-wasm';

import wasmSimdPath from './node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm';
import wasmSimdThreadedPath from './node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm';
import wasmPath from './node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm';

setWasmPaths({
  'tfjs-backend-wasm.wasm': wasmPath,
  'tfjs-backend-wasm-simd.wasm': wasmSimdPath,
  'tfjs-backend-wasm-threaded-simd.wasm': wasmSimdThreadedPath
});
```
