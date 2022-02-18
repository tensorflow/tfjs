# Optimization Topic - How to write a high performance GPU pipeline
This document shows how to write a high performance GPU pipeline with TF.js. A typical ML pipeline is composed of some ML steps (e.g. taking some inputs, running a model and getting outputs) and non-ML steps (e.g. image processing, rendering, etc.). General rule of thumb is that if the whole pipeline can run on GPU without downloading any data to CPU, it is usually much faster than a fragmented pipeline that requires data transfer between CPU and GPU. This additional time for GPU to CPU and CPU to GPU sync adds to the pipeline latency.

In the latest TF.js release (version 3.13.0), we provide a new tensor
API that allows data to stay in GPU. Our users are pretty faimilar with
`tensor.dataSync()` and `tensor.data()`, however both will download data from GPU to CPU. In the latest release, we have added `tensor.dataToGPU()`,
which returns a `GPUData` type.

The example below shows how to get the tensor texture by calling
`datatoGPU`. Also see this [example code](https://github.com/tensorflow/tfjs-examples/tree/master/gpu-pipeline) for details.

```javascript
// Getting the texture that holds the data on GPU.
// data has below fields:
// {texture, texShape, tensorRef}
// texture: A WebGLTexture.
// texShape: the [height, width] of the texture.
// tensorRef: the tensor associated with the texture.
//
// Here the tensor has a shape of [1, videoHeight, videoWidth, 4],
// which represents an image. In this case, we can specify the texture
// to use the image's shape as its shape.
const data =
      tensor.dataToGPU({customTexShape: [videoHeight, videoWidth]});

// Once we have the texture, we can bind it and pass it to the downstream
// webgl processing steps to use the texture.
gl.bindTexture(gl.TEXTURE_2D, data.texture);

// Some webgl processing steps.

// Once all the processing is done, we can render the result on the canvas.
gl.blitFramebuffer(0, 0, videoWidth, videoHeight, 0, videoHeight, videoWidth, 0, gl.COLOR_BUFFER_BIT, gl.LINEAR);

// Remember to dispose the texture after use, otherwise, there will be
// memory leak.
data.tensorRef.dispose();
```

The texture from `dataToGPU()` has a specific format. The data is densely
packed, meaning all four channels (r, g, b, a) in a texel is used to store
data. This is how the texture with tensor shape = [2, 3, 4] looks like:

```
    000|001|002|003   010|011|012|013   020|021|022|023

    100|101|102|103   110|111|112|113   120|121|122|123
```

So if a tensor's shape is `[1, height, width, 4]` or `[height, width, 4]`, the way we store the data aligns with how image is stored, therefore if the
downstream processing assumes the texture is an image, it doesn't need to
do any additional coordination conversion. Each texel is mapped to a [CSS pixel](https://developer.mozilla.org/en-US/docs/Glossary/CSS_pixel) in an image. However, if the tensor has other shapes, downstream
processing steps may need to reformat the data onto another texture. The `texShape` field has the `[height, width]` information of the texture, which can be used for transformation.
