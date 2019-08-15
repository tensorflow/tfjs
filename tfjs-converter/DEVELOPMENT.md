# Notes for TensorFlow.js-Converter Developers

## Python Development

There are some Python libraries, binary and tests in the `python/` directory.

It is recommended to do your Python development and testing in a
[virtualenv](https://virtualenv.pypa.io/en/stable/) or
[pipenv](https://docs.pipenv.org/).

As a prerequisite, install the following dependencies for python testing
```sh
cd python
pip install -r requirements.txt
```

For Python linter, install `pylint`, e.g.,
* `apt-get install -y pylint`

To run the Python linter:
```sh
cd python
pylint tensorflowjs
```

To run the python unit tests, there are two options. You can choose the one that
you prefer.

1. Run the tests using the `run-python-tests.sh` script:

   ```sh
   cd python
   ./run-python-tests.sh
   ```

2. Run the tests using Bazel. See bazel installation guide
   [here](https://docs.bazel.build/versions/master/install.html). Once bazel
   is installed, do:

   ```sh
   cd python
   bazel test tensorflowjs/...
   ```

Be sure to run the tests under **both** Python 2 and Python 3.

### Building and testing the tensorflowjs pip package

```sh
cd python

# You need to specify a folder where the pip wheel file will be stored, e.g.,
./build-pip-package.sh /tmp/my_tensorflowjs_pip

# If the script succeeds, you can use `pip install` to install the pip package:

pip install --force-reinstall \
  /tmp/my_tensorflowjs_pip/tensorflowjs-0.0.1-py2-none-any.whl
```

`build-pip-package.sh` provides a flag (`--test`) with which you can run a
test-on-install after building the pip package. Make sure you are using a
`virutalenv` or `pipenv` to avoid changing your base environmnet.

```sh
./build-pip-package.sh --test /tmp/my_tensorflowjs_pip
```
