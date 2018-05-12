## Development process

As a preparatory step, run `yarn` which installs all dev dependencies.

Before submitting a PR with a change, make sure the following
commands succeed:
* `yarn build` which compiles the project to ES5 Javascript.
* `yarn format` to format your code.
* `yarn lint` to check for linter errors.
* `yarn test` to run unit tests in Chrome and Firefox. Make sure all unit tests pass.

When you send a PR, the above commands will also run on [travis](https://travis-ci.org/tensorflow/tfjs-layers)
and show up as Github checks. If you see travis failing, click on the `Details`
link next to the check to open the travis log.

## Changing @tensorflow/tfjs-layers and testing @tensorflow/tfjs

Often we want to make a change in `tfjs-layers/core` and create a new
`tfjs` package that reflects that change. There is a 3-step initial process to
set this up. The instructions below are for `tfjs-layers`, but they should work
for developing `tfjs-core` if you replace `tfjs-layers` with `tfjs-core`.

1. In the `tfjs-layers` repo, run `yarn publish-local`. This builds the
project and publishes a new package in a local registry.

2. In the `tfjs` repo, run `yarn link-local @tensorflow/tfjs-layers`. This makes
`tfjs` depend on the locally published `tfjs-layers` package.

3. In the `tfjs` repo, run `yarn build-npm` to build a new npm package.

Every time you make a change in `tfjs-layers`, re-run:
- `yarn publish-local` in the `tfjs-layers` repo
- `yarn build-npm` in the `tfjs` repo to make a new package.

## Running integration tests

### tfjs2keras

This is an integration test that checks the models exported by tfjs-layers
can be loaded correctly by Keras in Python. To run this test, do:

```sh
yarn test-integ
```
