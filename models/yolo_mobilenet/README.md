# MobileNet model

This package contains a standalone MobileNet model for detection - You Only Look Once (YOLO).

## Installation
You can use this as standalone es5 bundle like this:

```html
<script src="https://unpkg.com/deeplearn-mobilenet"></script>
```

Or you can install it via npm for use in a TypeScript / ES6 project.

```sh
npm install deeplearn-yolo_mobilenet --save-dev
```

## Usage

Check out [demo.html](https://github.com/PAIR-code/deeplearnjs/blob/master/yolo_mobilenet/demo.html)
for an example with ES5.

To run the demo, use the following:

```bash
cd models/yolo_mobilenet

npm run prep
npm run build

# Starts a webserver, navigate to localhost:8000/demo.html.
python -m SimpleHTTPServer
```
