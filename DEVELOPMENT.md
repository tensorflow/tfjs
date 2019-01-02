## Development

This repository contains only the logic and scripts that combine
two packages:
- [TensorFlow.js Core](https://github.com/tensorflow/tfjs-core),
  a flexible low-level API, formerly known as *deeplearn.js*.
- [TensorFlow.js Layers](https://github.com/tensorflow/tfjs-layers),
  a high-level API which implements functionality similar to
  [Keras](https://keras.io/).

To develop:
- `tfjs-core`, see [this doc](https://github.com/tensorflow/tfjs-core/blob/master/DEVELOPMENT.md).
- `tfjs-layers` see [this doc](https://github.com/tensorflow/tfjs-layers/blob/master/DEVELOPMENT.md).
- `tfjs` with locally modified `tfjs-layers` (or `tfjs-core`) see [this section](https://github.com/tensorflow/tfjs-layers/blob/master/DEVELOPMENT.md#changing-tensorflowtfjs-layers-and-testing-tensorflowtfjs) in [tfjs-layers/DEVELOPMENT.md](https://github.com/tensorflow/tfjs-layers) repo.

## For repository owners: commit style guide

When merging commits into master, it is important to follow a few conventions
so that we can automatically generate release notes and have a uniform commit
history.

1. When you squash and merge, the default commit body will be all of the
commits on your development branch (not the PR description). These are usually
not very useful, so you should remove them, or replace them with the PR
description.

2. Release notes are automatically generated from commits. We have introduced a
few tags which help sort commits into categories for release notes:

- FEATURE (when new functionality / API is added)
- BREAKING (when there is API breakage)
- BUG (bug fixes)
- PERF (performance improvements)
- DEV (development flow changes)
- DOC (documentation changes)
- SECURITY (security changes)

These tags correspond to GitHub labels which are automatically prepended to your PR description.
Please add the appropriate labels to your PR.

A typical commit may look something like:

```
Subject: Add tf.toPixels. (#900)
Body:
FEATURE

tf.toPixels is the inverse of tf.fromPixels, writing a tensor to a canvas.

```

This will show up under "Features" as:
- Add tf.toPixels. (#900). Thanks, @externalcontributor.


You can also use multiple tags for the same commit if you want it to show up in
two sections. You can add clarifying text on the line of the tags.

You can add clarifying messages on the line of the tag as well.

For example:

```
Subject: Improvements to matMul. (#900)
Body:

FEATURE Add transpose bits to matmul.
PERFORMANCE Improve matMul CPU speed by 100%.
```

This will show up under "Features" as:
- Add transpose bits to matmul (Improvements to matMul.) (#900). Thanks, @externalcontributor.

This will also show up under "Performance" as:
- Improve matMul CPU speed by 100%. (Improvements to matMul.) (#900). Thanks, @externalcontributor.
