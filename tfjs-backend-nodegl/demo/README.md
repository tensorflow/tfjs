# MobileNet tfjs-backend-nodegl Demo

*This is a very early demo to show how tfjs-backend-nodegl can be used for headless WebGL acceleration.*

To run this demo, perform the following:

1. Move into `tfjs-backend-nodegl` (parent directory of this demo folder):
```sh
$ cd tfjs-backend-nodegl
```

2. Build package and compile TypeScript:
```sh
$ yarn && yarn tsc
```

3. Move into the demo directory:
```sh
$ cd demo
```

4. Prep and build demo:
```sh
$ yarn
```

5. Run demo:
```sh
$ node run_mobilenet_inference.js dog.jpg
```

Expected output:
```sh
$ node run_mobilenet_inference.js dog.jpg
Platform node has already been set. Overwriting the platform with [object Object].
  - gl.VERSION: OpenGL ES 3.0 (ANGLE 2.1.0.9512a0ef062a)
  - gl.RENDERER: ANGLE (Intel Inc., Intel(R) Iris(TM) Plus Graphics 640, OpenGL 4.1 core)
  - Loading model...
  - Mobilenet load: 6450.763924002647ms
  - Coldstarting model...
  - Mobilenet cold start: 297.92842200398445ms
  - Running inference (100x) ...
  - Mobilenet inference: (100x) : 35.75772546708584ms
```
