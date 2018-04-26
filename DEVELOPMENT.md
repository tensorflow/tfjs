# Development

This repository contains a union of tensorflow/tfjs-layers and tensorflow/tfjs-core.
Development happens inside of each of the individual repositories.

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

If no tag is specified, it will be put under a "MISC" tag.

You can tag a commit with these tags by putting the tag at the beginning of a
new line. Any text after the tag on that line will show up in the release notes
next to the subject.

A typical commit may look something like:

```
Subject: Add tf.toPixels. (#900)
Body:

tf.toPixels is the inverse of tf.fromPixels, writing a tensor to a canvas.

FEATURE
```

This will show up under "Features" as:
- Add tf.toPixels. (#900). Thanks, @externalcontributor.


You can also use multiple tags for the same commit if you want it to show up in
two sections. You can add clarifying text on the line of the tags.


You can add clarifying messages on the line of the tag if they

. For examples:

```
Subject: Improvements to matMul. (#900)
Body:

FEATURE Adds transpose bits to matmul.
PERFORMANCE Improves matMul CPU speed by 100%.
```

This will show up under "Features" as:
- Adds transpose bits to matmul (Improvements to matMul.) (#900). Thanks, @externalcontributor.

This will also show up under "Performance" as:
- Improves matMul CPU speed by 100%. (Improvements to matMul.) (#900). Thanks, @externalcontributor.
