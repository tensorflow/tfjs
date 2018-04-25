## Core (0.7.1 ==> 0.8.4)
- update version to 0.8.4. Thanks @dsmilkov.
- Fix build issue where module-import code gets dropped from the bundle ([#987](https://github.com/tensorflow/tfjs-core/pull/987)). Thanks @dsmilkov.

> `backend_cpu.ts` and `backend_webgl.ts` are never included from `index.ts` even though there is
> `export {WebGLTimingInfo} from './kernels/backend_webgl';`. That export is just a Typescript interface, and gets compiled away in the es5 code.
- Update version to 0.8.3 ([#986](https://github.com/tensorflow/tfjs-core/pull/986)). Thanks @dsmilkov.
- test_util still depends on jasmine typings because of `DoneFn`. This fixes it ([#985](https://github.com/tensorflow/tfjs-core/pull/985)). Thanks @dsmilkov.
- Update to 0.8.2. ([#984](https://github.com/tensorflow/tfjs-core/pull/984))
- Add yarn "link-local" and yarn "publish-local" ([#983](https://github.com/tensorflow/tfjs-core/pull/983)). Thanks @dsmilkov.

> These commands are better alternatives to `yarn link` and resolve caching and de-duplication problems:
> - `yarn publish-local` replaces `yarn link`
> - `yarn link-local PCK_NAME` replaces `yarn link PCK_NAME`
- Stop recording to the tape while a kernel runs ([#980](https://github.com/tensorflow/tfjs-core/pull/980)). Thanks @dsmilkov.

> Some kernels might call higher-level ops to work around limitations. Because of this, we need to stop recording to the tape while the kernel runs.
> 
> Also use built-in inversesqrt in GLSL
- Add a test in batchNorm where mean,var,scale,offset are high-dim and not symmetric ([#979](https://github.com/tensorflow/tfjs-core/pull/979)). Thanks @dsmilkov.

> This tests reveals problems in other backends (e.g. node).
> 
> Also handle a bad state in tensor.get()
- Add quotemark rule to tslint. ([#975](https://github.com/tensorflow/tfjs-core/pull/975)). Thanks @ry.
- Remove double quotes in imagenet_classes.ts. ([#977](https://github.com/tensorflow/tfjs-core/pull/977))
- test_util shouldn't depend on jasmine. ([#974](https://github.com/tensorflow/tfjs-core/pull/974)). Thanks @ry.
- Modify tf.split signature to preserve rank. ([#973](https://github.com/tensorflow/tfjs-core/pull/973)). Thanks @iansimon.
- Bump version to 0.8.1 ([#971](https://github.com/tensorflow/tfjs-core/pull/971))
- Fix some issues in batchNormalization gradient ([#970](https://github.com/tensorflow/tfjs-core/pull/970))

> * Fix some issues in batchNormalization gradient
> 
> Two issues fixed:
> 1. The outgoing gradient (`dy`) was previously not used.
> 2. The ordering of reduce sum and multiplication was wrong in
>    some of the gradients previously.
> 
> Also tested with BatchNormalization training in tfjs-layers:
> https://github.com/tensorflow/tfjs-layers/pull/139/files
- Add tf.erf op ([#951](https://github.com/tensorflow/tfjs-core/pull/951)). Thanks @Lewuathe.

> [As TensorFlow does](https://www.tensorflow.org/api_docs/python/tf/erf), TensorFlow.js can provide `erf` ops to calculate error function. The actual algorithm calculates the value approximately with elementary functions. See [Approximation with elementary functions](https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions).
> 
> <img width="977" alt="screen shot 2018-04-14 at 19 19 49" src="https://user-images.githubusercontent.com/1713047/38767243-d11d3616-4018-11e8-9fbe-d247f020e5f9.png">
- Assert arguments to ops are Tensors. ([#967](https://github.com/tensorflow/tfjs-core/pull/967))
- Require shape in tensor2d/3d/4d(flatValues, shape) ([#969](https://github.com/tensorflow/tfjs-core/pull/969)). Thanks @dsmilkov.

> **Bug**
> `tensor2d([1, 2, 3, 4])` returns Tensor of rank 1 with shape `[4]`. Likewise for `tensor3d` and `tensor4d`
> 
> **Solution**
> `tensor2d(flatValues)` throws an error requiring shape to be provided explicitly by the user.
- Bump version to 0.8.0 ([#968](https://github.com/tensorflow/tfjs-core/pull/968))
- Add tf.movingAverage. ([#963](https://github.com/tensorflow/tfjs-core/pull/963))
- Add slice ergonomics. ([#964](https://github.com/tensorflow/tfjs-core/pull/964)). Thanks @ry.
- Fix integer division. ([#966](https://github.com/tensorflow/tfjs-core/pull/966))
- Add tf.resizeNearestNeighbor ([#955](https://github.com/tensorflow/tfjs-core/pull/955)). Thanks @Lewuathe.

> There is [a use case](https://github.com/tensorflow/tfjs/issues/148) to use `tf.resizeNearestNeighbor`. 
> As well as `tf.resizeBilinear`, we can add this image resizing algorithm.
> 
> Fixes https://github.com/tensorflow/tfjs/issues/159
- Update to 0.7.2. ([#962](https://github.com/tensorflow/tfjs-core/pull/962))
- Implement floordiv for integer division & update CONTRIBUTING.md. ([#960](https://github.com/tensorflow/tfjs-core/pull/960))
- Add code examples to jsdocs. ([#899](https://github.com/tensorflow/tfjs-core/pull/899)). Thanks @nbardy.
- Clean up demo scripts. ([#949](https://github.com/tensorflow/tfjs-core/pull/949)). Thanks @ManrajGrover.
- Finish gradient for tf.pow ([#954](https://github.com/tensorflow/tfjs-core/pull/954)). Thanks @jgartman.
- Add gradients for tf.batchNormalization4d ([#959](https://github.com/tensorflow/tfjs-core/pull/959))

> Add gradients for x, mean, variance, offset and scale.
> 
> A step toward fulfilling: tensorflow/tfjs#20
- Finish gradient for tf.matMul ([#957](https://github.com/tensorflow/tfjs-core/pull/957)). Thanks @jgartman.
- Make CPU matmul 100x faster ([#961](https://github.com/tensorflow/tfjs-core/pull/961)). Thanks @piscisaureus.
- Fix error message in clipByValue ([#953](https://github.com/tensorflow/tfjs-core/pull/953)). Thanks @mlajtos.
- Fix typo in tsdoc ([#952](https://github.com/tensorflow/tfjs-core/pull/952)). Thanks @Lewuathe.
- Align API with TF ([#956](https://github.com/tensorflow/tfjs-core/pull/956)). Thanks @dsmilkov.

> Aligns the backend API and functionality (NaN propagation, dtype strictness, kernel signatures) with TensorFlow Python.
> 
> - Remove `backend.minPool` since TF doesn't have it.
> - Remove `normRegion` param in `localResponseNormalization` kernel since TF doesn't support it.
> - Remove `leakyRelu`, `prelu` and `preluDer` from the backend and implement using higher-level ops, aligning with TF Python.
> - Make backend.multinomial take `logits` instead of `probabilities` and `normalized: boolean` param for backwards compatibility.
> - `argMin` and `argMax` take single `axis: number` instead of `axes: number[]`
> - Change `eluDer(x: T): T` signature to `eluDer(dy: T, y: T): T` to align with TF.
> - Change NaN behavior of `max/avgPool` and `conv2d` to align with TF.
> - Change `avgPool` out of bounds (padding) behavior to align with TF
> - Require `indices` in `oneHot` and `gather` to be of dtype `int32`
> 
> Fixes tensorflow/tfjs#195
- Add tf.logSigmoid and tf.softplus and gradients ([#916](https://github.com/tensorflow/tfjs-core/pull/916)). Thanks @jgartman.
- Remove deeplearnjs references ([#948](https://github.com/tensorflow/tfjs-core/pull/948)). Thanks @ManrajGrover.
- Align webgl and cpu backend to be closer to TF ([#947](https://github.com/tensorflow/tfjs-core/pull/947)). Thanks @dsmilkov.

> - Remove NaN support for tensors of dtype `bool` and `int32`
> - Remove related tests
> - Make `fromPixels` test run only in the browser
> - Remove NaN propagation for min/max/argMin/argMax/compare/logical ops to align with TF
> - Speed up randomNormal tests
> - Minimize usage of isNaN checks in WebGL shaders
- remove old demos, move benchmarks to integration_tests ([#943](https://github.com/tensorflow/tfjs-core/pull/943)). Thanks @tafsiri.
- Make core tests reusable ([#926](https://github.com/tensorflow/tfjs-core/pull/926)). Thanks @dsmilkov.

> The goal of this PR is to make the core tests reusable across different backends.
> 
> - `describeWithFlags` instead of list of features, now takes constraints that need to be satisfied in order for the test to run.
> - `ALL_ENVS` becomes `{}`, which means there are no constraints.
> - `WEBGL_ENVS` becomes `{BACKEND: 'webgl'}` which means run this test only when the `BACKEND` environment feature is `webgl`. Likewise for `CPU_ENVS`.
> - Adds the following user-facing API:
>   - `tf.test_util.setBeforeAll`/`setBeforeEach(f)`
>   - `tf.test_util.setAfterAll`/`setAfterEach(f)`
>   - `tf.test_util.setTestEnvFeatures(listOfFeatures)`. An example list: `[{BACKEND: 'custom'}]`
> - Constraints a few core tests to run only in WebGL, since they are using browser-specific API.
> - `ENV.registerBackend()` now takes an optional priority number (defaults to 1), which is used by `getBestBackendType` to find the best backend.
> - Splits `toPixels` tests into browser-specific (which use canvas) and non-browser specific
> 
> Also fixes a bug with vertex buffer attribute binding, which allows us to re-create webgl backends in beforeAll, instead of beforeEach, speeding up our tests by 3x.
- Fix snippet comments for sum and mean ([#946](https://github.com/tensorflow/tfjs-core/pull/946)). Thanks @mlajtos.
- Add tf.toPixels. ([#941](https://github.com/tensorflow/tfjs-core/pull/941))
- Rename dl -> tf ([#944](https://github.com/tensorflow/tfjs-core/pull/944)). Thanks @ManrajGrover.
- Array Ops: Fixes tf.clone documentation ([#942](https://github.com/tensorflow/tfjs-core/pull/942)). Thanks @ManrajGrover.

## Layers (0.4.0 ==> 0.5.3)
- Update version to 0.5.3. Thanks @dsmilkov.
- Update to 0.5.2. ([#151](https://github.com/tensorflow/tfjs-layers/pull/151))
- Update to 0.5.1 and update the layers tests to use jasmine_util. ([#150](https://github.com/tensorflow/tfjs-layers/pull/150))
- Improve `DEVELOPMENT.md` ([#148](https://github.com/tensorflow/tfjs-layers/pull/148)). Thanks @dsmilkov.

> - Add `yarn link-local` and `yarn publish-local` to substitute `yarn link` and avoid problems with duplication of packages and caching
> - Simplify `DEVELOPMENT.md` and add text for the newly added commands.
- Bump version to 0.5.0 ([#147](https://github.com/tensorflow/tfjs-layers/pull/147))
- Support training of BatchNormalization layer ([#139](https://github.com/tensorflow/tfjs-layers/pull/139))

> * Support training of BatchNormalization layer
> 
> Fixes: https://github.com/tensorflow/tfjs/issues/20
> 
> * chaser
> 
> * Add test case for 3D
> 
> * Add Python code for 3D test case
> 
> * Update tfjs-core dep version to 0.8.1
- add @doc to getLayer ([#145](https://github.com/tensorflow/tfjs-layers/pull/145))
- Expose tf.metrics; Add binaryCrossentropy metric ([#144](https://github.com/tensorflow/tfjs-layers/pull/144))

> * Expose tf.metrics; Add binaryCrossentropy metric
> 
> Fixes: https://github.com/tensorflow/tfjs/issues/208
> 
> * Add test for one-hot binary case
- Export Shape type alias: tf.Shape ([#142](https://github.com/tensorflow/tfjs-layers/pull/142)). Thanks @ry.
- Add quotemark to tslint. ([#143](https://github.com/tensorflow/tfjs-layers/pull/143)). Thanks @ry.
- Introduce Serializable base class ([#140](https://github.com/tensorflow/tfjs-layers/pull/140)). Thanks @ericdnielsen.

> * Introduce Serializable base class
> 
> * Remove constructor.name from layer related code
> 
> * Remove constructor.name from Tensor related call sites
> 
> * Add doc-strings for Serializable's methods.
- Update switch-tfjs-core-version.sh (switch-deeplearn-version.sh) ([#138](https://github.com/tensorflow/tfjs-layers/pull/138))

> * Update switch-tfjs-core-version.sh (switch-deeplearn-version.sh)
> 
> * Rename the file to reflect repo change.
> * Replace deeplearn.js with tfjs-core in the script.
> 
> * Remove obsolete comment in doc string.
- Tests use external api ([#137](https://github.com/tensorflow/tfjs-layers/pull/137))

> * conv tests use (some of the) external api
> 
> * Update convolutional_test to use external version of SymbolicTensor
> 
> * more move to use external API
> 
> * more layers
> 
> * Finished going through all the test files.
> 
> * fixup factory function types
> 
> * always import as tfl
- Typo: Corrected typo thresohldedReLU -> thresholdedReLU ([#136](https://github.com/tensorflow/tfjs-layers/pull/136)). Thanks @xam-ps.
- Increase tfjs-core dependency version to 0.8.0 ([#135](https://github.com/tensorflow/tfjs-layers/pull/135))
- Export Reshape layer as tf.layers.reshape ([#134](https://github.com/tensorflow/tfjs-layers/pull/134)). Thanks @shanqing-cai.

> * Export Reshape layer as tf.layers.reshape
> 
> Fixes: https://github.com/tensorflow/tfjs/issues/204
> 
> * Fix formatting
- Fix a bug in custom-loss support ([#131](https://github.com/tensorflow/tfjs-layers/pull/131))

> * Fix a bug in custom-loss support
> 
> * losses.get() now does the correct thing when a custom loss function is
>   passed in
> 
> Fixes: https://github.com/tensorflow/tfjs/issues/198
> 
> * Respond to reviewer comments
- Add unit tests to cover deserialization of Merge layer subtypes ([#132](https://github.com/tensorflow/tfjs-layers/pull/132))

> * Add unit tests to cover deserialization of Merge layer subtypes
- Update dependency tfjs-core to 0.7.2 ([#130](https://github.com/tensorflow/tfjs-layers/pull/130)). Thanks @dsmilkov.
- Add some print-outs to a code snippet ([#133](https://github.com/tensorflow/tfjs-layers/pull/133)). Thanks @shanqing-cai.

> * Add some print-outs to a code snippet
> 
> The code snippet for Sequential.fit() did not have any
> printed messages previously, which was confusing.
> 
> This PR fixes that.
- Remove underscore as a dependency ([#129](https://github.com/tensorflow/tfjs-layers/pull/129))

> * Remove _.range
> 
> * Remove _.isNull and _.isUndefined
> 
> * Remove _.sortBy
> 
> * Remove _.max and _.times
> 
> * Remove _.isEqual
> 
> * Remove _.contains
> 
> * Remove _.uniq and _.unique
> 
> * Remove _.any and _.every
> 
> * Remove _.zip
> 
> * Remove _.pairs
> 
> * Remove _.has
> 
> * Remove _.shuffle
> 
> * Remove _.keys
> 
> * Remove _.isEmpty
> 
> * Remove _.flatten
> 
> * Remove underscore from package.json
- Fix serialization of Sequential ([#127](https://github.com/tensorflow/tfjs-layers/pull/127))

> * Fix serialization of Sequential
> 
> Make the config an Array, not a dict with a field 'layers'. This is
> for compatibility with Python Keras.
> 
> Fixes: https://github.com/tensorflow/tfjs/issues/190
- Cast indices to int as needed. ([#128](https://github.com/tensorflow/tfjs-layers/pull/128)). Thanks @nkreeger.

> * Cast indices to int as needed.
> 
> * Fix clang-format.
- Bump version to 0.4.1 ([#126](https://github.com/tensorflow/tfjs-layers/pull/126))
- Let Conv1D take [number] for dilationRate ([#125](https://github.com/tensorflow/tfjs-layers/pull/125))
- Issues Template: Remove label issues from template ([#124](https://github.com/tensorflow/tfjs-layers/pull/124)). Thanks @ManrajGrover.
- Check the equality of tfjs version in dev and peer dependencies ([#123](https://github.com/tensorflow/tfjs-layers/pull/123))

> * Check the equality of tfjs version in dev and peer dependencies
> * Disable tslint warning