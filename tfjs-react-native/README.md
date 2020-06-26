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

These instructions (and this library) **assume that you are generally familiar with [react native](https://facebook.github.io/react-native/) development**.

## Expo compatibility

This library relies on [expo-gl](https://github.com/expo/expo/tree/master/packages/expo-gl) and [expo-gl-cpp](https://github.com/expo/expo/tree/master/packages/expo-gl-cpp). Thus you must use a version of React Native that is supported by Expo.

Some parts of tfjs-react-native are not compatible with _managed expo apps_. You must use the bare workflow (or just plain react native) if you want to use the following functionality:
 - Loading local models using [bundleResourceIO](https://js.tensorflow.org/api_react_native/latest/#bundleResourceIO). You can instead load models from a webserver.

### Step 1. Create your react native app.

You can use the [React Native CLI](https://facebook.github.io/react-native/docs/getting-started) or [Expo](https://expo.io/).

On macOS (to develop iOS applications) You will also need to use CocoaPods to install these dependencies.

### Step 2: Install dependencies

Note that if you are using in a managed expo app the install instructions may be different.

  - Install and configure [react-native-unimodules](https://github.com/unimodules/react-native-unimodules) (can be skipped if in an expo app)
  - Install and configure [expo-gl-cpp](https://github.com/expo/expo/tree/master/packages/expo-gl-cpp) and [expo-gl](https://github.com/expo/expo/tree/master/packages/expo-gl)
  - Install and configure [expo-gl](https://github.com/expo/expo/tree/master/packages/expo-gl)
  - Install and configure [expo-gl-cpp](https://github.com/expo/expo/tree/master/packages/expo-gl-cpp) and [expo-gl](https://github.com/expo/expo/tree/master/packages/expo-gl)
  - Install and configure [expo-camera](https://www.npmjs.com/package/expo-camera)
  - Install and configure [async-storage](https://github.com/react-native-community/async-storage)
  - Install and configure [react-native-fs](https://www.npmjs.com/package/react-native-fs)
  - **Install @tensorflow/tfjs** - `npm install @tensorflow/tfjs`
  - **Install @tensorflow/tfjs-react-native** - `npm install @tensorflow/tfjs-react-native`


> After this point, if you are using Xcode to build for ios, you should use a ‘.workspace’ file instead of the ‘.xcodeproj’

### Step 3: Configure [Metro](https://facebook.github.io/metro/en/)

This step is only needed if you want to use the [bundleResourceIO](https://js.tensorflow.org/api_react_native/latest/#bundleResourceIO) loader.

Edit your `metro.config.js` to look like the following. Changes are noted in
the comments below.

```js
const { getDefaultConfig } = require('metro-config');
module.exports = (async () => {
  const defaultConfig = await getDefaultConfig();
  const { assetExts } = defaultConfig.resolver;
  return {
    resolver: {
      // Add bin to assetExts
      assetExts: [...assetExts, 'bin'],
    }
  };
})();
```

### Step 4: Test that it is working

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

You can take a look at [`integration_rn59/App.tsx`](integration_rn59/App.tsx) for an example of what using tfjs-react-native looks like. In future we will add an example to the [tensorflow/tfjs-examples](https://github.com/tensorflow/tfjs-examples) repository.
The [Webcam demo folder](integration_rn59/components/webcam) has an example of a style transfer app.

![style transfer app initial screen](images/rn-styletransfer_1.jpg)
![style transfer app initial screen](images/rn-styletransfer_2.jpg)
![style transfer app initial screen](images/rn-styletransfer_3.jpg)
![style transfer app initial screen](images/rn-styletransfer_4.jpg)


## API Docs

[API docs are available here](https://js.tensorflow.org/api_react_native/latest/)

## Compatibility with TFJS models

Many [tfjs-models](https://github.com/tensorflow/tfjs-models) use web APIs for rendering or input, these are not generally compatible with React Native, to use them you generally need to **feed a tensor** into the model and do any rendering of the model output with react native components. If there is no API for passing a tensor into a [tfjs-model](https://github.com/tensorflow/tfjs-models), feel free to file a GitHub issue.

## Debugging and reporting errors

When reporting bugs with tfjs-react-native please include the following information:

  - Is the app created using expo? If so is it a managed or bare app?
  - Which version of react native and the dependencies in the install instructions above are you using?
  - What device(s) are you running on? Note that not all simulators support webgl and thus may not work with tfjs-react-native.
  - What error messages are you seeing? Are there any relevant messages [in the device logs](https://reactnative.dev/docs/debugging#accessing-console-logs)?
  - How could this bug be reproduced? Is there an example repo we can use to replicate the issue?
