# tfjs-layers benchmarks

To run the benchmark script, first set up your environment.

(You may wish to set up Python the requirements in a virtual environment using
[pipenv](https://github.com/pypa/pipenv) or [virtualenv](https://virtualenv.pypa.io))

```
pip install tensorflowjs
```

Once the development environment is prepared, execute the build script from the root of tfjs-layers.

```
./scripts/build-benchmarks-demo.sh
```

The script will construct a number of Keras models in Python and benchmark their training using the TensorFlow backend.  When it is complete, it will bring up a
local HTTP server.  Navigate to the local URL spcecified in stdout to bring up
the benchmarks page UI.  There will be a button to begin the JS side of the
benchmarks.  Clicking the button will run through and time the same models, now
running in the browser.

Once complete, the models' `fit()` and `predict()` costs are listed in a table.

Prese Ctl-C to end the http-server process.
