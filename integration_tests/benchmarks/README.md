# TensorFlow.js: Layers API Benchmarks

These benchmarks measure the inference and training speed of models of
varying size and architecture, comparing TensorFlow.js in the browser with
Python Keras backed by TensorFlow.

NOTE: This benchmark requires that you have TensorFlow and Keras installed in your
Python environment since it compares timing in Python to timing in the browser.

To launch the benchmarks, first make sure you have all the required Python
packages installed:

```sh
cd integration_tests/benchmarks
pip install -r requirements.txt
```

Then, run the benchmarks in Python and build the benchmark page in JavaScript:

```sh
./build-benchmarks.sh
```

Once the step above has been done, you can skip the Python benchmark part in
future runs by using the `--skip_py_benchmarks` flag, i.e.,

```sh
./build-benchmarks.sh --skip_by_benchmarks
```
