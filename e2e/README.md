# Usage

This package contains integration and e2e tests for TensorFlow.js.

## To run the tests in local:

### Filter tests by tag:

```js
export TAGS=#SMOKE,#REGRESSION,#GOLDEN
yarn test
```

### Filter tests by grep:

```js
yarn test --grep cpu
```

## Add new tests:

When creating new test, add at least one tag to the test description.

Supported tags:

- **SMOKE**:
  Smoke tests should be light weight. Run in every PR and nightly
  builds. Criteria for smoke test is that the test should run fast and only
  test critical CUJ.
- **REGRESSION**:
  Regression tests compare results across backends, previous
  builds, with other platform, etc. Run in nightly builds. Need additional
  steps to run in local.
- **GOLDEN**:
  Golden tests validate pretrained model outputs against the prebuilt golden
  outputs. Need additional steps to run in local.

To add a new tag: Extend the TAGS list in integration_tests/util
