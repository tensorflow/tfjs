---
layout: page
order: 2
---
# Getting the best performance

This guide is intended to help developers make applications using deeplearn.js
more performant.

## Chrome Developer Tools - Performance Tab

One of the most powerful tools for debugging your deeplearn.js application is
the performance tab in Chrome developer tools. There is much more information
on the details of the performance tab
[here](https://developers.google.com/web/tools/chrome-devtools/evaluate-performance/),
but we will focus on how to use it in a deeplearn.js application.

Below is a performance trace of one of the core demos. The two rows to pay
attention to are the "Main" thread and the "GPU" process:

![Chrome Developer Tools - Performance tab](devtools-performance.png "Chrome Developer Tools - Performance tab")

The "Main" thread shows browser activity on the main CPU UI thread. This is
where JavaScript is executed, and where page layout / some painting happens.

The "GPU" process shows the activity of the GPU process. For compute intensive
applications, we want this row to be solid green, which indicates we're maximally
utilizing the GPU. If we are constantly introducing interlocks between the CPU
and the GPU, the GPU process may look choppy as the GPU is waiting for more work
to be scheduled. There are exceptions to this rule, for example if there isn't a
lot of work to be done.


### Understanding CPU / GPU interlocks and `NDArray.data()`

The most common thing that will cause a performance issue is a blocking
`gl.readPixels` call on the main thread. This is the underlying WebGL call
that downloads NDArrays from a WebGL texture to the CPU. This function returns
a `Float32Array` with the underlying values from the `WebGLTexture`-backed
`NDArray`.

`gl.readPixels` is a CPU-blocking call which waits for the GPU to finish its
execution pipeline until the given NDArray is available, and then downloads it.
This means that the time you see in the performance tab on the "Main" thread
row corresponding to the `gl.readPixels` call is not actually the time it takes
to download, but the time the UI thread is blocking and waiting for the result
to be ready.

By blocking the UI thread with the `gl.readPixels` call, we don't allow the
browser's UI thread to do anything else during that time. This includes layout,
painting, responding to user events, and practically all interactivity. This
will cause serious jank issues that make the webpage unusable.

To mitigate this, you should always use `NDArray.data()` to resolve
values to the CPU. This function returns a `Promise<Float32Array>` that
only calls `gl.readPixels` when the GPU process has completed all the work
necessary to resolve the values of the given `NDArray`. This means that the UI
thread can do other things while it is waiting for the GPU work to be done,
mitigating jank issues. This is why, in general, you should *avoid*
`NDArray.dataSync()`, which simply calls `gl.readPixels` directly.

> You should *not* call other `NDArrayMath` functions while waiting for the
`NDArray.data()`s `Promise` to resolve as this may introduce a stall when we call
the underlying `gl.readPixels` command. A common pattern is to only call
`NDArray.data()` at the end of your main application loop, and only call the loop
function again inside the resolved `Promise`. Alternatively, you can use
`await NDArray.data()` to ensure the next lines that enqueue GPU programs wait
for data to be ready.


## Memory leaks & dl.tidy

Preventing memory leaks in applications that may have an infinite loop
(for example, reading from the webcam and making a prediction) is critical for
performance.

When operations are used, you should wrap them in a `dl.tidy(f)` call where
`f` is the function that does the job. All allocated memory in the function
will get disposed at the end of the closure, except for the returned value.

A function is passed to the function closure, `keep()`.

`keep()` ensures that the NDArray passed to keep will not be cleaned up
automatically when the closure ends.

```ts
import * as dl from 'deeplearn';

let output;

dl.tidy(keep => {
  // CORRECT: By default, dl tracks all NDArrays that are constructed.
  const a = dl.Scalar.new(2);

  // CORRECT: By default, dl tracks all outputs of operations.
  const c = a.exp().neg();

  // CORRECT: d is tracked by the parent function.
  const d = dl.tidy(() => {
    // CORRECT: e will get cleaned up when this inner function ends.
    const e = dl.Scalar.new(3);

    // CORRECT: The result of this function is tracked. Since it is the
    // return value, it will not get cleaned up after this function ends.
    // It will instead, be tracked in the parent function.
    return e.mul(e);
  });

  // CORRECT, BUT BE CAREFUL: The output of tanh() will be tracked
  // automatically, however we can call keep() on it so that it will be kept
  // when the function ends. That means if you are not careful about calling
  // output.dispose() some time later, you might introduce a texture memory
  // leak. A better way to do this would be to return this value as a return
  // value of a function so that it gets tracked in a parent scope.
  output = keep(d.tanh());
});
```

> More technical details: When WebGL textures go out of scope in JavaScript,
they don't get cleaned up automatically by the browser's garbage collection
mechanism. This means when you are done with an NDArray that is GPU-resident,
it must manually be disposed some time later. If you forget to manually call
`ndarray.dispose()` when you are done with an NDArray, you will introduce
a texture memory leak, which will cause serious performance issues.
If you use `dl.tidy(f)`, where `f` is your function that does work, any NDArrays
created directly or indirectly will be automatically cleaned up after `f` ends.


> If you want to do manual memory management and not use dl.tidy(), you can.
This is not recommended, but is useful for `NDArrayMathCPU` since CPU-resident
memory will get cleaned up automatically by the JavaScript garbage collector.

### TODO(nsthorat|smilkov): How to track down memory leaks

## Debug mode

Another way to monitor your application is by calling
`NDArrayMath.enableDebugMode()`. In debug mode, we will profile every
`NDArrayMath` function that gets called while the application is running,
logging the command, the wall time in milliseconds, the rank, shape and the
size. We also will check the activations for NaNs and throw an exception as soon as a NaN is introduced.

![NDArrayMath.enableDebugMode](debugmode.png "NDArrayMath.enableDebugMode")

This can give you a sense for which operations are the bottleneck in
your application, and which activations are using large amounts of memory.

> Keep in mind debug mode slows down your application as we download values
after every operation to check for NaNs. You should not use this mode when
you deploy your application to the world.
