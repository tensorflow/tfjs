# Emscripten installation

Install emscripten [here](https://emscripten.org/docs/getting_started/downloads.html).

Make sure the "emsdk" directory is in your home directory at ~/emsdk.

# Building

```sh
yarn build
```

# Testing

```sh
yarn test
```

# Deployment
```sh
./scripts/build-npm.sh
npm publish
```
