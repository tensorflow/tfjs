# Development

This file will document some of the differences from the regular developement workflow in [DEVELOPMENT.md](../DEVELOPMENT.md). You should read that document first to get familiar with typical TensorFlow.js development workflow.

Development and testing for tfjs-react-native is somewhat different from the packages like tfjs-core or tfjs-layers for a few reasons:
- __Dependency on having a physical mobile device to run__: While the CPU backend can run in a simulator, the WebGL one requires running on a physical device. So most of the time you will want to test something using a mobile device connected to your computer.
- __No browser or node environment__: We are running JavaScript outside of a browser and outside of node. We thus have to make sure we don't include things that depend on those two environments.


## Key Terms & Caveats

These are a few key terms/technologies to be familiar with that are different from what we use for web or node.js development.

- [React Native](https://facebook.github.io/react-native/) — This is the framework that this package targets.
- [Metro](https://facebook.github.io/metro/) — This is the bundler used to create the JavaScript bundle that is loaded into the native app by react native.
  - The bundle needs to be created at 'compile time' thus all imports/requires need to be resolved. Thus _dynamic_ `import`s/`require`s are __statically resolved__. So you cannot exclude a require with a conditional in JS code.
  - Since tfjs does dynamic `require`'s of certain node libraries that are not present in react native, files that do that need to be excluded from the metro build process. For end users, this is documented in the [README](../README.md), but it also happens in `integration_rn59/prep_tests.ts`.
  - Metro does not play well with symlinks, so if you are trying to develop against a local build of tfjs, copy the dist folder into the app's node_modules as appropriate. Do not use yalc.
- [.ipa](https://en.wikipedia.org/wiki/.ipa) & [.apk](https://en.wikipedia.org/wiki/Android_application_package) — These are the formats for the final native bundle that is put on an iOS and Android device. They are created by their respective dev tools, [XCode](https://developer.apple.com/xcode/) and [Android Studio](https://developer.android.com/studio).

## Testing

There are three categories of tests that are run for this package.

### tfjs-react-native unit tests

Unit tests are written for the package functionality and are imported into a react native application to run on device. They are run together with the integration tests described below to simplify automation.

### tfjs-core unit tests running on device in react native

Unit tests from tfjs-core are imported into a react native application and run as integration tests. This allows testing of tfjs-core against actual devices through the RN bridge and against expo's WebGL bindings.

Because these are part of an app to run them you must compile and run the integration_rn59 of the target device. There is a button in that app to start the unit tests.

This is _automated in CI_ and runs on:
 - Changes to tfjs-core: [Tests will be run against HEAD of tfjs-core](../tfjs-core/cloudbuild.yml)
 - Changes to tfjs-react-native: [Tests will be run against the **published** version](./cloudbuild.yml) of tfjs on npm that is references in `integration_rn59/package.json`

### Other integration tests

The integration_rn59 app also contains some other tests and sanity checks. These can be run manually. Future on device integration tests should also be incorporated into this app.

## CI Testing Infrastructure

Integration tests on CI have a few moving pieces, the basic workflow is as follows.

1. A native app ([.apk](https://storage.googleapis.com/tfjs-rn/integration-tests/app-debug.apk)) is built manually and stored on GCP. This app doesn't need to change unless a new native dependency is added.
2. To update this `apk` run `yarn update-api`. This will update what is stored on GCP and also sync it to Browserstack.
3. The app is also synced periodically from GCP to BrowserStack as BrowserStack caches it for 30 days from the last update. This is done with a cloud function (sync_reactnative) that is triggered via cloud scheduler.
4. On PRs a Google Cloud Build builder will trigger a browserstack test (using browserstack app automate), and serve the JS bundle to the device running in browserstack using metro. The tests are designed to create a tunnel between the native device and the cloud builder machine.
5. The tests complete on browserstack and results are reported back to Google Cloud Build which are reported back to GitHub.
