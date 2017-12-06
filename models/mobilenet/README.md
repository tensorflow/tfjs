# MobileNet model

This package contains a standalone MobileNet model for classification.

## Installation
You can use this as standalone es5 bundle like this:

```html
<script src="https://unpkg.com/deeplearn-mobilenet"></script>
```

Or you can install it via npm for use in a TypeScript / ES6 project.

```sh
npm install deeplearn-mobilenet --save-dev
```

## Usage

Check out [demo.html](https://github.com/PAIR-code/deeplearnjs/blob/master/models/mobilenet/demo.html)
for an example with ES5.

To run the demo, use the following:

```bash
cd models/mobilenet

npm run prep
npm run build

# Starts a webserver, navigate to localhost:8000/demo.html.
python -m SimpleHTTPServer
```
