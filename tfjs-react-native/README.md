# Platform Adapter for React Native

This package provides a TensorFlow.js platform adapter for react native. It
provides GPU accelerated execution of TensorFlow.js supporting all major modes
of tfjs usage, include:
  - Support for both model inference and training
  - GPU support with WebGL via expo-gl.
  - Support for loading models pretrained models (tfjs-models) from the web.
  - IOHandlers to support loading models from asyncStorage and models
    that are compiled into the app bundle.

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
- Install @tensorflow/tfjs-react-native - `npm install @tensorflow/tfjs-react-native`

### Step 5: Install and configure other peerDependencies

- Install and configure [async-storage](https://github.com/react-native-community/async-storage)
- Install and configure [react-native-fs](https://www.npmjs.com/package/react-native-fs)
- Install and configure [expo-camera](https://www.npmjs.com/package/expo-camera)

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

[API docs are available here](https://js.tensorflow.org/api_react_native/latest/)
