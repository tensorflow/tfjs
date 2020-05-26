# Usage

This package contains integration and e2e tests for TensorFlow.js.

To run the tests in local:

##Filter tests by tag:
```js
yarn test --tags #SMOKE
```
Supported tags:
- #SMOKE: Smoke tests should be light weight. Run in every PR and nightly
    builds. Criteria for smoke test is that the test should run fast and only
    test critical CUJ.

- #REGRESSION: Regression tests compare results across backends, previous
    builds, with other platform, etc. Run in nightly builds. Need additional
    steps to run in local.

Some tags may requires additional steps to run, reference `scripts/test-ci.sh`
for additional steps.

To add a new tag: Extend the TAGS list in integration_tests/util

**Note:** When creating new test, prefer add at least one tag to the test
description. Otherwise, the test will only run when no --tags is specified.

##Filter tests by grep:
```js
yarn test --grep cpu
```
