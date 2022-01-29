# Optimization Topic - How to write a high performance GPU pipeline.
This document shows how to write a high performance GPU pipeline with TF.js. What we often encounter in a GPU pipeline is that, there are often
some ML steps (e.g. taking some inputs, running a model and get outputs)
and non-ML steps (e.g. image processing, rendering, etc.). The generarl
idea is that if the whole pipeline can run on GPU without ever syncing to
CPU, it is usually much faster than a broken pipeline where some steps
have to download data from GPU to CPU to process and then upload to GPU
again. This additional time for GPU to CPU and CPU to GPU sync adds to
the pipeline latency.

In the latest TF.js release (version 3.13.0), we provide a new tensor
API that allows data to stay in GPU. Our users are pretty faimilar with
`tensor.dataSync()` and `tensor.data()`, however both will download data from GPU to CPU. In the latest release, we have added `tensor.dataToGPU()`,
which returns a `GPUData` type. This object contains a `texture` field, a
`texShape` and a `tensorRef` field. The `texture` field holds a `WebGLTexture`, `texShape` holds the `[height, width]` informatioin of a
texture and the `tensorRef` field holds the tensor that contains this texture and related tensor information.

The example below is pretty much self explained. We get the tensor texture by calling `dataToGPU()`. Here the tensor has a shape of
[1, videoHeight, videoWidth, 4], which represents an image. In this case, we can specify the texture shape to be the same as the image. We will discuss tensors with other shapes in next section. Once we have the texture, we can bind it and pass it to the downstream webgl processing steps to use the texture. Once all the processing is done, we can render
the result on the canvas. Remember to dispose the texture after use, otherwise, there will be memory leak. See this [example code](https://github.com/tensorflow/tfjs-examples/tree/master/gpu-pipeline) for details.

```javascript
// Getting the texture that holds the data on GPU.
const data =
      tensor.dataToGPU({customTexShape: [videoHeight, videoWidth]});
gl.bindTexture(gl.TEXTURE_2D, data.texture);

// Some webgl processing steps.

// Render on canvas.
gl.blitFramebuffer(0, 0, videoWidth, videoHeight, 0, videoHeight, videoWidth, 0, gl.COLOR_BUFFER_BIT, gl.LINEAR);

// Dispose the texture.
data.tensorRef.dispose();
```

The texture from `dataToGPU()` has a specific format. The data is densely
packed, meaning all four channels (r, g, b, a) in a texel is used to store
data. This is how the texture with tensor shape = [2, 3, 4] looks like:

```
    000|001   010|011   020|021
    -------   -------   -------
    002|003   012|013   022|023

    100|101   110|111   120|121
    -------   -------   -------
    102|103   112|113   122|123
```

So if a tensor's shape is [1, height, width, 4] or [height, width, 4], the
way we store the data aligns with how image is stored, therefore if the
downstream processing assumes the texture is an image, it doesn't need to
do any additional coordination conversion. Each texel is mapped to a CSS pixel in an image. However, if the tensor has other shapes, downstream
processing steps need to compute where in the texture to get the right data. The `texShape` field has the `[height, width]` information of the
texture, which can be used to compute to get the coordination of the texel
to get data.
