# Usage

This package contains integration and e2e tests for TensorFlow.js.

To run the tests in local:
```js
yarn test
```

Filter tests by tag:
```js
yarn test --tags #SMOKE,#LAYERS
```
Supported tags: #SMOKE, #REGRESSION, #LAYERS

To add new tag: Extend the TAGS list in integration_tests/util

Filter tests by grep:
```js
yarn test --grep cpu
```
