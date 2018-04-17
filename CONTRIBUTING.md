# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Adding functionality

One way to ensure that your PR will be accepted is to add functionality that
has been requested in Github issues. If there is something you think is
important and we're missing it but does not show up in Github issues, it would
be good to file an issue there first so we can have the discussion before
sending us a PR.

In general, we're trying to add functionality when driven by use-cases instead of
adding functionality for the sake of parity with TensorFlow python. This will
help us keep the bundle size smaller and have less to maintain especially as we
add new backends.

### Adding an op

When adding ops to the library and deciding whether to write a kernel
implementation in [backend.ts](https://github.com/tensorflow/tfjs-core/blob/master/src/kernels/backend.ts),
be sure to check out the TensorFlow ops list [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/ops.pbtxt).
This list shows the kernels available for the TensorFlow C API. To ensure that
we can bind to this with node.js, we should ensure that our backend.ts
interface matches ops in the TensorFlow C API.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

We require unit tests for most code, instructions for running our unit test
suites are in the documentation.
