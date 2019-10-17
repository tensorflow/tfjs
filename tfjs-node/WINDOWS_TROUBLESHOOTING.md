# TensorFlow.js Node.js bindings Windows troubleshooting.

The tfjs-node package uses the [node-gyp](https://github.com/nodejs/node-gyp) package to handle cross-compiling the C++ code required to bind TensorFlow to Node.js. This cross platform solution can be somewhat tricky on Windows platforms. This guide helps reference solutions that have been triaged through Issues on the main [tfjs repo](https://github.com/tensorflow/tfjs).

## Ensure Python 2.x is installed

Currently, node-gyp requires Python 2.x to work properly. If Python 3.x is installed, you will see build failures. Also double check your python version `python --version` and update the Windows `$PATH` as needed.

## 'The system cannot find the patch specified' Exceptions

This can happen for a variety of reasons. First, to inspect what is missing either `cd node_modules/@tensorflow/tfjs-node` or clone the [tensorflow/tfjs repo](https://github.com/tensorflow/tfjs).

After `cd`'ing or cloning, run the following command (you might need node-gyp installed globablly `npm install -g node-gyp`):

```sh
node-gyp configure --verbose
```

### Missing `python2`

From the verbose output, if you see something like:

```sh
gyp verb check python checking for Python executable "python2" in the PATH
gyp verb `which` failed Error: not found: python2
```

This means that node-gyp expects a 'python2' exe somewhere in `%PATH%`. Try running this command from an Admin (elevated privilaged prompt):

You can try running this from an Adminstrative prompt:

```sh
$ npm --add-python-to-path='true' --debug install --global windows-build-tools
```

### Something else?

If another missing component shows up - please file an issue on [tensorflow/tfjs](https://github.com/tensorflow/tfjs/issues/new) with the output from the `node-gyp configure --verbose` command.

## msbuild.exe Exceptions

Check the full stack trace from `npm install` (or `yarn`) command. If you see something like:

```
gyp ERR! stack Error: C:\Program Files (x86)\MSBuild\14.0\bin\msbuild.exe failed with exit code: 1
```

You might need to install the system tools manually. This can be done via:

```
npm install -g --production windows-build-tools
```

If that still does not work - try re-running the command above in a privileged shell.
