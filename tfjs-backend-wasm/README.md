# Emscripten installation

Install the Emscripten SDK (version 1.39.1):

```sh
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install 1.39.1
./emsdk activate 1.39.1
```

# Prepare the environment

Before developing, make sure the environment variable `EMSDK` points to the
emscripten directory (e.g. `~/emsdk`). Emscripten provides a script that does
the setup for you:

Cd into the emsdk directory and run:

```sh
source ./emsdk_env.sh
```

For details, see instructions
[here](https://emscripten.org/docs/getting_started/downloads.html#installation-instructions).

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
