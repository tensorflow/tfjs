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

# Running MobileNet

### Option 1: with bundles

#### 1) Build core as a bundle
```sh
cd tfjs-core
./scripts/build-npm.sh
```

### 2) Build the WASM backend as a bundle
```sh
cd tfjs-backend-wasm
./scripts/build-npm.sh
```

### 3) Load the scripts
```html
<script src="../tfjs-core/dist/tf-core.js"></script>
<script src="https://unpkg.com/@tensorflow/tfjs-converter"></script>
<script src="dist/tf-backend-wasm.js"></script>
```

### 4) Load and run mobilenet with the WASM backend
```js
<script>
async function go() {
  // Set the backend to WASM.
  tf.setBackend('wasm');
  // Wait for the wasm backend to be ready.
  await tf.ready();

  const img = tf.browser.fromPixels(document.getElementById('img'))
      .resizeBilinear([224, 224])
      .expandDims(0)
      .toFloat();

  const model = await tf.loadGraphModel(
    'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2',
    {fromTFHub: true});

  model.predict(img).print();
}
go();
</script>
```

### Option 2: with a local NPM package via yalc

#### 1) Build core as a local yalc package
```sh
cd tfjs-core
yarn publish-local
```

#### 2) Build the WASM backend as a bundle
```sh
cd tfjs-backend-wasm
./scripts/build-npm.sh
```

#### 3) Build the WASM backend as a local yalc package.
```sh
cd tfjs-backend-wasm
yarn publish-local
```

### 4) Load the packages
Make sure you yalc link the packages from your new project:
```sh
yarn yalc link @tensorflow/tfjs-core
yarn yalc link @tensorflow/tfjs-backend-wasm
```


```js
import * as tfc from '@tensorflow/tfjs-core';
<script src="../tfjs-core/dist/tf-core.js"></script>
<script src="https://unpkg.com/@tensorflow/tfjs-converter"></script>
<script src="dist/tf-backend-wasm.js"></script>
```

### 3b) Load the packages
```js
import * as tfc from '@tensorflow/tfjs-core';

<script src="../tfjs-core/dist/tf-core.js"></script>
<script src="https://unpkg.com/@tensorflow/tfjs-converter"></script>
<script src="dist/tf-backend-wasm.js"></script>
```

### 3) Load and run MobileNet

```html
<script src="../tfjs-core/dist/tf-core.js"></script>
<script src="https://unpkg.com/@tensorflow/tfjs-converter"></script>
<script src="dist/tf-backend-wasm.js"></script>

<!-- Load your image-->

<img id="img" src="coffee.jpg" crossorigin="anonymous">



```
