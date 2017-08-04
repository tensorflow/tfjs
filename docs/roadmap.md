---
layout: page
order: 1000
---
# Roadmap

This page outlines some of the projects we wish to happen in the near future.
These are projects that we would love to see the open source community
contribute to.

# More devices

**deeplearn.js** targets WebGL 1.0 devices with the `OES_texture_float`
extension and targets WebGL 2.0 devices. However, we have turned off mobile,
Safari, and Firefox for demos - they should work with some minor changes
to the WebGL API.

## Optimizers

Currently, **deeplearn.js** only has an SGD optimizer, however the optimizer
interface is generic enough to support new optimizers. We would love to see RMSProp,
Adagrad, Adadelta, Adam, and Adamax.

## Logical sampling

When writing custom shader programs in **deeplearn.js**, the author must sample
textures in 2D physical texture space. This means that if an shader program
operates on an `Array3D`, it must manually convert between logical 3D space and
physical 2D texture space. Since shader programs are a little tricky to debug,
this makes shader programs error-prone.

We have started on "logical sampling", that is, introducing functions and a
shader compiler that allows shaders to sample in logical space through a utility
function. This means we can store higher dimensional NDArrays in 2D textures in
whatever shape we want to ensure minimal reshapes when chaining operations.

Currently, matmul is the only GPU shader program that uses the new shader compiler
and logical sampling, but it should serve as a guide for the way shader programs
should be migrated, and how new shader programs should be written.

## Batch as the outer dimension

**deeplearn.js** at the shader level only supports operations with a batch size
of 1, whereas most other machine learning libraries use the batch size as an
outer dimension. This is usually okay for many applications, though it can be
restrictive when models are ported.

As part of the new shader compiler and helper functions to do logical sampling,
we now can introduce batching as an outer dimension of operations.

## Automatic TensorFlow to deeplearn.js

Currently we support dumping weights from a TensorFlow checkpoint into a format
that can be imported into **deeplearn.js**, however the developer must then
recreate the model in **deeplearn.js** and use the weights from that checkpoint.

We plan on building a way to port models directly from TensorFlow to
**deeplearn.js** automatically from a `GraphDef`.

## Dynamic batching

Dynamic batching, which allows training with explicitly defining a graph, but
instead simply analyzing the forward mathematical operations and differentiating
that dynamic computation graph, is a popular method for training models.

We can implement dynamic batching by doing it at the NDArrayMath layer. When
mathematical methods are called, we can record operations that were called and
automatically differentiate when requested.

## Recurrence in training

**deeplearn.js** doesn't currently support recurrence as top level
functionality during training, however we do support arbitrary data flow graphs,
so recurrence should be straight forward to implement.
