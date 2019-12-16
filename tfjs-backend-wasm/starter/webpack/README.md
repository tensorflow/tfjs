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

Then we obtain the final serving path of the `tfjs-backend-wasm.wasm` file
that was shipped on NPM, and use `setWasmPath` to let the library know the
serving location:

```ts
import {setWasmPath} from '@tensorflow/tfjs-backend-wasm';
import wasmPath from '../node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm';
setWasmPath(wasmPath);
```
