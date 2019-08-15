# Platform Adapter for React Native

Status: __Early development__. This is still an unpublished experimental package.

## Adapter Docs

TODO

## Setting up a React Native app with tfjs-react-native

These instructions assume that you are generally familiar with [react native](https://facebook.github.io/react-native/) developement. This library has only been tested with React Native 0.58.X & 0.59.X. React Native 0.60 is not supported.

### Step 1. Create your react native app.

You can use the [React Native CLI](https://facebook.github.io/react-native/docs/getting-started) or [Expo](https://expo.io/). This library relies on a couple of dependencies from the Expo project so it may be convenient to use expo but is not mandatory.

On macOS (to develop iOS applications) You will also need to use Cocoapods to install these dependencies.

### Step 2: Install expo related libraries

Depending on which workflow you used to set up your app you will need to install different dependencies.

- React Native CLI App
  - Install and configure [react-native-unimodules](https://github.com/unimodules/react-native-unimodules)
  - Install and configure [expo-gl-cpp](https://github.com/expo/expo/tree/master/packages/expo-gl-cpp) and [expo-gl](https://github.com/expo/expo/tree/master/packages/expo-gl)
- Expo Bare App
  - Install and configure [expo-gl-cpp](https://github.com/expo/expo/tree/master/packages/expo-gl-cpp) and [expo-gl](https://github.com/expo/expo/tree/master/packages/expo-gl)
- Expo Managed App
  - Install and configure [expo-gl](https://github.com/expo/expo/tree/master/packages/expo-gl)


Note if in a _managed_ expo application these libraries should be present and you should be able to skip this step.

Install and configure [react-native-unimodules](https://github.com/unimodules/react-native-unimodules)
Install and configure [expo-gl-cpp](https://github.com/expo/expo/tree/master/packages/expo-gl-cpp) and [expo-gl](https://github.com/expo/expo/tree/master/packages/expo-gl)

> After this point, if you are using XCode to build for ios, you should use a ‘.workspace’ file instead of the ‘.xcodeproj’

### Step 3: Configure [Metro](https://facebook.github.io/metro/en/)

Edit your `metro.config.js` to look like the following. Changes are noted in
the comments below.

```js
// Change 1
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
    // Change 3
    blacklistRE: blacklist([/platform_node/])
  },
};
```


### Step 4: Install TensorFlow.js and tfjs-react-native

- Install @tensorflow/tfjs - `npm install @tensorflow/tfjs`
- Install @tensorflow/tfjs-react-native - coming soon

### Step 5: Test that it is working

TODO: Add some sample code.

For now take a look at `integration_rn59/App.tsx` for an example of what using tfjs-react-native looks like.

### Optional Steps

If you want use the `AsyncStorageHandler` to save and load models, add [async-storage](https://github.com/react-native-community/async-storage) to your project.

