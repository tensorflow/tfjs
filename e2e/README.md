# Usage

This package contains integration and e2e tests for TensorFlow.js.

To run the tests in local:
```js
yarn test
```

For tests that need to access `integration_tests/test_data`, start a local
server in the test_data folder:
```
cd integration_tests/test_data
yarn http-server
```

##Filter tests by tag:
```js
yarn test --tags #SMOKE,#REGRESSION
```
Supported tags:
- #SMOKE: Smoke tests should be light weight. Run in every PR and nightly
    builds.

- #REGRESSION: Regression tests compare results across backends, previous
    builds, with other platform, etc. Run in nightly builds.

To add a new tag: Extend the TAGS list in integration_tests/util

**Note:** When creating new test, prefer add at least one tag to the test
description. Otherwise, the test will only run when no --tags is specified.

##Filter tests by grep:
```js
yarn test --grep cpu
```
