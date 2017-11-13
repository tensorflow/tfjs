---
layout: page
order: 1000
---
# Roadmap

This page outlines some of the projects we wish to happen in the near future.
These are projects that we would love to see the open source community
contribute to.

## Automatic TensorFlow to deeplearn.js

Currently we support dumping weights from a TensorFlow checkpoint into a format
that can be imported into **deeplearn.js**, however the developer must then
recreate the model in **deeplearn.js** and use the weights from that checkpoint.

We plan on building a way to port models directly from TensorFlow to
**deeplearn.js** automatically from a `GraphDef`.

## Decoupling NDArray from storage mechanism

Currently, `NDArray`s are tightly coupled to their underlying storage. We will
be decoupling the `NDArray` object from where it is actually stored, and add
global tracking to all `NDArray`s so that we don't need to explicitly `track` them
inside of a `math.scope()`.

This also means `scope` will become a top level method.

## Eager mode

To train or get gradients, you must use our `Graph` layer. We will be
adding an Eager execution mode in the near term future, similar to
[TensorFlow Eager](https://research.googleblog.com/2017/10/eager-execution-imperative-define-by.html).

This will vastly simplify debugging as training will just be a call to
`NDArrayMath.backward()`.

## Model zoo

We started working on a model zoo, which can be found
[here](https://github.com/PAIR-code/deeplearnjs/tree/master/models). They can
be used independently through npm.

We want to see this built out.

## Top level math functions

We will be adding math operations at the top level, like this: `dl.matMul`
instead of having to construct `NDArrayMath` objects directly. This will make
code look much cleaner and similar to well-known libraries like TensorFlow and NumPy.

## deeplearn.js Canvas (aka Playground)

We recently launched [deeplearn.js canvas](https://deeplearnjs.org/demos/playground/index.html),
which allows you to play with deeplearn.js without having to clone our
repository or compile TypeScript.

We will add a button for "saving" soon, so these can be shared. We will also
move tutorials over.
