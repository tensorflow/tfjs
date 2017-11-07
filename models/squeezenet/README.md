# SqueezeNet model

This package contains a standalone SqueezeNet model.

## Installation
You can use this as standalone es5 bundle like this:

```html
<script src="https://unpkg.com/deeplearn-squeezenet"></script>
```

Or you can install it via npm for use in a TypeScript / ES6 project.

```sh
npm install deeplearn-squeezenet --save-dev
```

## Usage

Check out [demo.html](https://github.com/PAIR-code/deeplearnjs/blob/master/models/squeezenet/demo.html)
for an example with ES5.

To run the demo, use the following:

```bash
cd models/squeezenet

npm run prep
npm run build

# Starts a webserver, navigate to localhost:8000/demo.html.
python -m SimpleHTTPServer
```
