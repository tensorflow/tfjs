# Starter project for Parcel

A minimal starter project using Parcel and the WASM backend.
See [Using Bundlers](../../README.md#using-bundlers) section in the main readme
for details on how we serve the WASM backend on NPM.

To serve the `.wasm` file, we use the `parcel-plugin-static-files-copy` plugin
and added the following to the `package.json`, which copies any `.wasm` file to
the Parcel output directory:

```json
...,
"staticFiles": {
    "staticPath": "./node_modules/@tensorflow/tfjs-backend-wasm/dist",
    "excludeGlob": ["**/!(*.wasm)"]
  },
...
```
