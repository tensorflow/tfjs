# Cartoonizer demo

This demo "cartoonizes" a set of pictures and your webcam feed by using the
`tfjs-tflite` package and the CartoonGAN model described in this
[blog post][blog].

Here is the [live demo][live demo] you can try out. For best performance, enable
the "WebAssembly SIMD support" and "WebAssembly threads support" in
`chrome://flags/`.

<img src="screenshot.png" alt="demo" style="max-width: 930px; width: 100%;"/>

## Run the demo locally

Build the dependencies.

```sh
$ yarn build-deps
$ yarn
```

Run the demo locally
```sh
$ yarn watch
```


[blog]: https://blog.tensorflow.org/2020/09/how-to-create-cartoonizer-with-tf-lite.html
[live demo]: https://storage.googleapis.com/tfweb/demos/cartoonizer/index.html
[gsutil]: https://cloud.google.com/storage/docs/gsutil_install
