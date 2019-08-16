TypeScript Interfaces describing the Keras JSON format
------------------------------------------------------

This directory contains a description of the current Keras JSON serialization
format, in the form of TypeScript interfaces.  The intent is that any valid
Keras JSON file can be parsed in a type-safe manner using these types.

The Keras JSON format originated in the Python Keras implementation.  The basic
design is that the format mirrors the Python API.  Each class instance in a
Python model is serialized as a JSON object containing the class name and its
serialized constructor arguments.

Here, we provide a type called `*Serialization` to describe the on-disk JSON
representation for each class.  It always provides a `class_name` and a `config`
representing the constructor arguments required to reconstruct a given instance.

The constructor arguments may be primitives, arrays of primitives, or plain
key-value dictionaries, in which case the JSON serialization is straightforward.

If a constructor argument is another object, then it is represented by a nested
`*Serialization`.  This structure is illustrated below:

    FooSerialization {
      class_name: 'Foo';
      config: {
        bar: string;
        baz: number[];
        qux: QuxSerialization;
      }
    }

Deserializing such a nested object configuration requires recursively
deserializing any object arguments, and finally calling the top-level
constructor using the reconstructed object arguments.

In general this means that deserialization is purely tree-like, so instances
cannot be reused.  (The deserialization code for Models is an exception to this
principle, because it allows Layers to refer to each other in order to describe
a DAG).

As a consequence of this design, our deserialization code requires an `*Args`
type mirroring each of the `*Serialization` types here.  `*Args` types represent
the actual arguments passed to a constructor, after any nested objects have been
deserialized.  For instance, the above `FooSerialization` will be resolved into
 `FooArgs` like this:

     FooArgs {
       bar: string;
       baz: number[];
       qux: Qux;
     }

which can then be passed to the `Foo` constructor.
