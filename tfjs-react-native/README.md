# Platform Adapter for React Native

This package provides a TensorFlow.js platform adapter for react native. It
provides GPU accelerated execution of TensorFlow.js supporting all major modes
of tfjs usage, include:
  - Support for both model inference and training
  - GPU support with WebGL via expo-gl.
  - Support for loading models pretrained models (tfjs-models) from the web.
  - IOHandlers to support loading models from asyncStorage and models
    that are compiled into the app bundle.

## Status
This package is currently an **alpha release**. We welcome react native developers
to try it and give us feedback.

## Setting up a React Native app with tfjs-react-native

These instructions **assume that you are generally familiar with [react native](https://facebook.github.io/react-native/) developement**.

### Step 1. Create your react native app.

You can use the [React Native CLI](https://facebook.github.io/react-native/docs/getting-started) or [Expo](https://expo.io/). This library relies on a couple of dependencies from the Expo project so it may be convenient to use expo but is not mandatory.

On macOS (to develop iOS applications) You will also need to use CocoaPods to install these dependencies.

### Step 2: Install expo related libraries

Depending on which workflow you used to set up your app you will need to install different dependencies.

- React Native CLI App
  - Install and configure [react-native-unimodules](https://github.com/unimodules/react-native-unimodules)
  - Install and configure [expo-gl-cpp](https://github.com/expo/expo/tree/master/packages/expo-gl-cpp) and [expo-gl](https://github.com/expo/expo/tree/master/packages/expo-gl)
- Expo Bare App
  - Install and configure [expo-gl-cpp](https://github.com/expo/expo/tree/master/packages/expo-gl-cpp) and [expo-gl](https://github.com/expo/expo/tree/master/packages/expo-gl)
- Expo Managed App
  - Install and configure [expo-gl](https://github.com/expo/expo/tree/master/packages/expo-gl)

> If you are in a _managed_ expo application these libraries should be present and you should be able to skip this step.

> After this point, if you are using Xcode to build for ios, you should use a ‘.workspace’ file instead of the ‘.xcodeproj’

### Step 3: Configure [Metro](https://facebook.github.io/metro/en/)

Edit your `metro.config.js` to look like the following. Changes are noted in
the comments below.

```js
// Change 1 (import the blacklist utility)
const blacklist = require('metro-config/src/defaults/blacklist');

module.exports = {
  transformer: {
    getTransformOptions: async () => ({
      transform: {
        experimentalImportSupport: false,
        inlineRequires: false,
      },
    }),
  },
  resolver: {
    // Change 2 (add 'bin' to assetExts)
    assetExts: ['bin', 'txt', 'jpg'],
    sourceExts: ['js', 'json', 'ts', 'tsx', 'jsx'],
    // Change 3 (add platform_node to blacklist)
    blacklistRE: blacklist([/platform_node/])
  },
};
```

### Step 4: Install TensorFlow.js and tfjs-react-native

- Install @tensorflow/tfjs - `npm install @tensorflow/tfjs`
- Install @tensorflow/tfjs-react-native - `npm install @tensorflow/tfjs-react-native@alpha`

### Step 5: Install and configure other peerDependencies

- Install and configure [async-storage](https://github.com/react-native-community/async-storage)
- Install and configure [react-native-fs](https://www.npmjs.com/package/react-native-fs)

### Step 6: Test that it is working

Before using tfjs in a react native app, you need to call `tf.ready()` and wait for it to complete. This is an **async function** so you might want to do this in a `componentDidMount` or before the app is rendered.

The example below uses a flag in the App state to indicate that TensorFlow is ready.


```js
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';

export class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isTfReady: false,
    };
  }

  async componentDidMount() {
    // Wait for tf to be ready.
    await tf.ready();
    // Signal to the app that tensorflow.js can now be used.
    this.setState({
      isTfReady: true,
    });
  }


  render() {
    //
  }
}

```

After gathering feedback in the alpha release we will add an example to the [tensorflow/tfjs-examples](https://github.com/tensorflow/tfjs-examples) repository.

For now you can take a look at [`integration_rn59/App.tsx`](integration_rn59/App.tsx) for an example of what using tfjs-react-native looks like.
The [Webcam demo folder](integration_rn59/components/webcam) has an example of a style transfer app.

![style transfer app initial screen](images/rn-styletransfer_1.jpg)
![style transfer app initial screen](images/rn-styletransfer_2.jpg)
![style transfer app initial screen](images/rn-styletransfer_3.jpg)
![style transfer app initial screen](images/rn-styletransfer_4.jpg)


## API Docs

`tfjs-react-native` exports a number of utility functions:

### asyncStorageIO(modelKey: string)

```js
async function asyncStorageExample() {
  // Define a model
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 5, inputShape: [1]}));
  model.add(tf.layers.dense({units: 1}));
  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

  // Save the model to async storage
  await model.save(asyncStorageIO('custom-model-test'));
  // Load the model from async storage
  await tf.loadLayersModel(asyncStorageIO('custom-model-test'));
}
```

The `asyncStorageIO` function returns an io handler that can be used to save and load models
to and from AsyncStorage.

### bundleResourceIO(modelArchitecture: Object, modelWeights: number)

```js
const modelJson = require('../path/to/model.json');
const modelWeights = require('../path/to/model_weights.bin');
async function bundleResourceIOExample() {
  const model =
      await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));

  const res = model.predict(tf.randomNormal([1, 28, 28, 1])) as tf.Tensor;
}
```

The `bundleResourceIO` function returns an IOHandler that is able to **load** models
that have been bundled with the app (apk or ipa) at compile time. It takes two
parameters.

1. modelArchitecture: This is a JavaScript object (and notably not a string). This is
   because metro will automatically resolve `require`'s for JSON file and return parsed
   JavaScript objects.

2. modelWeights: This is the numeric id returned by the metro bundler for the binary weights file
   via `require`. The IOHandler will be able to load the actual data from the bundle package.

`bundleResourceIO` only supports non sharded models at the moment. It also cannot save models. Though you
can use the asyncStorageIO handler to save to AsyncStorage.

### decodeJpeg(contents: Uint8Array, channels?: 0 | 1 | 3)

```js
const image = require("path/to/img.jpg");
const imageAssetPath = Image.resolveAssetSource(image);
const response = await fetch(imageAssetPath.uri, {}, { isBinary: true });
const rawImageData = await response.arrayBuffer();

const imageTensor = decodeJpeg(rawImageData);
```

**returns** a tf.Tensor3D of the decoded image.

Parameters:

1. contents: raw bytes of the image as a Uint8Array
1. channels: An optional int that indicates whether the image should be loaded as RBG (channels = 3), Grayscale (channels = 1), or autoselected based on the contents of the image (channels = 0). Defaults to 3. Currently only 3 channel RGB images are supported.

### fetch(path: string, init?: RequestInit, options?: tf.io.RequestDetails)

tfjs react native exports a custom fetch function that is able to correctly load binary files into
`arrayBuffer`'s. The first two parameters are the same as regular [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API). The 3rd paramater is an optional custom `options` object, it currently has one option

- options.isBinary: A boolean indicating if this is request for a binary file.

This is needed because the response from `fetch` as currently implemented in React Native does not support the `arrayBuffer()` call.
