# TensorFlow SavedModel Import Demo

This demo imports the
[MobileNet v1.0](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)
model for inference in the browser. The model was pre-converted to TensorFlow.js
format and hosted on Google Cloud, using the steps in
the repo's [README.md](../../README.md).

The following commands will start a web server on `localhost:1234` and open
a browser page with the demo.

```bash
cd demo # If not already in the demo directory.
yarn # Installs dependencies.
yarn mobilenet # Starts a web server and opens a page. Also watches for changes.
```
