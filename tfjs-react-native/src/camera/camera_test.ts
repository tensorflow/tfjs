/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';
import {test_util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';
import {ExpoWebGLRenderingContext, GLView} from 'expo-gl';

import {RN_ENVS} from '../test_env_registry';

import {detectGLCapabilities, fromTexture, toTexture} from './camera';

async function createGLContext(): Promise<ExpoWebGLRenderingContext> {
  return GLView.createContextAsync();
}

const expectArraysEqual = test_util.expectArraysEqual;

let gl: ExpoWebGLRenderingContext;

describeWithFlags('toTexture', RN_ENVS, () => {
  beforeAll(async () => {
    if (gl == null) {
      gl = await createGLContext();
    }
  });

  it('should not throw', async () => {
    const height = 2;
    const width = 2;
    const depth = 4;

    const inTensor: tf.Tensor3D =
        tf.truncatedNormal([height, width, depth], 127, 40, 'int32');

    let texture: WebGLTexture;
    expect(async () => {
      texture = await toTexture(gl, inTensor);
    }).not.toThrow();

    expect(texture instanceof WebGLTexture);
  });

  it('should roundtrip succesfully', async () => {
    const height = 2;
    const width = 2;
    const depth = 4;

    const inTensor: tf.Tensor3D =
        tf.truncatedNormal([height, width, depth], 127, 40, 'int32');
    const texture = await toTexture(gl, inTensor);

    const outTensor = fromTexture(
        gl, texture, {width, height, depth}, {width, height, depth});

    expectArraysEqual(await inTensor.data(), await outTensor.data());
    expectArraysEqual(inTensor.shape, outTensor.shape);
  });

  it('throws if tensor is not int32 dtype', async () => {
    const height = 2;
    const width = 2;
    const depth = 4;

    const floatInput: tf.Tensor3D =
        tf.truncatedNormal([height, width, depth], 127, 40, 'float32');

    expectAsync(toTexture(gl, floatInput)).toBeRejected();
  });

  it('throws if tensor is not a tensor3d dtype', async () => {
    const batch = 2;
    const height = 2;
    const width = 2;
    const depth = 4;

    const oneDInput: tf.Tensor1D =
        tf.truncatedNormal([height], 127, 40, 'int32');
    //@ts-ignore
    expectAsync(toTexture(gl, oneDInput)).toBeRejected();

    const twoDInput: tf.Tensor2D =
        tf.truncatedNormal([height, width], 127, 40, 'int32');
    //@ts-ignore
    expectAsync(toTexture(gl, twoDInput)).toBeRejected();

    const fourDInput: tf.Tensor4D =
        tf.truncatedNormal([batch, height, width, depth], 127, 40, 'int32');
    //@ts-ignore
    expectAsync(toTexture(gl, fourDInput)).toBeRejected();
  });
});

describeWithFlags('fromTexture:nearestNeighbor', RN_ENVS, () => {
  let texture: WebGLTexture;
  let input: tf.Tensor3D;
  const inShape: [number, number, number] = [4, 4, 4];

  beforeAll(async () => {
    if (gl == null) {
      gl = await createGLContext();
    }

    input = tf.tensor3d(
        [
          [
            [200, 201, 202, 255],
            [190, 191, 192, 255],
            [180, 181, 182, 255],
            [170, 171, 172, 255],
          ],
          [
            [160, 161, 162, 255],
            [150, 151, 152, 255],
            [140, 141, 142, 255],
            [130, 131, 132, 255],
          ],
          [
            [120, 121, 122, 255],
            [110, 111, 112, 255],
            [100, 101, 102, 255],
            [90, 91, 92, 255],
          ],
          [
            [80, 81, 82, 255],
            [70, 71, 72, 255],
            [60, 61, 62, 255],
            [50, 51, 52, 255],
          ]
        ],
        inShape, 'int32');
  });

  beforeEach(async () => {
    texture = await toTexture(gl, input);
  });

  afterAll(() => {
    tf.dispose(input);
  });

  it('same size alignCorners=false', async () => {
    const output = fromTexture(
        gl,
        texture,
        {
          height: inShape[0],
          width: inShape[1],
          depth: inShape[2],
        },
        {
          height: inShape[0],
          width: inShape[1],
          depth: inShape[2],
        },
        {
          alignCorners: false,
          interpolation: 'nearest_neighbor',
        },
    );

    expectArraysEqual(await output.data(), await input.data());
    expectArraysEqual(output.shape, input.shape);
  });

  it('same size, alignCorners=true', async () => {
    const output = fromTexture(
        gl,
        texture,
        {
          height: inShape[0],
          width: inShape[1],
          depth: inShape[2],
        },
        {
          height: inShape[0],
          width: inShape[1],
          depth: inShape[2],
        },
        {
          alignCorners: true,
          interpolation: 'nearest_neighbor',
        },
    );

    expectArraysEqual(await output.data(), await input.data());
    expectArraysEqual(output.shape, input.shape);
  });

  it('smaller, resizeNearestNeighbor, same aspect ratio, alignCorners=false',
     async () => {
       const expectedShape: [number, number, number] = [2, 2, 4];
       const expected = tf.tensor3d(
           [
             [
               [200, 201, 202, 255],
               [180, 181, 182, 255],
             ],
             [
               [120, 121, 122, 255],
               [100, 101, 102, 255],
             ]
           ],
           expectedShape, 'int32');

       const output = fromTexture(
           gl,
           texture,
           {
             height: inShape[0],
             width: inShape[1],
             depth: inShape[2],
           },
           {
             height: expectedShape[0],
             width: expectedShape[1],
             depth: expectedShape[2],
           },
           {alignCorners: false, interpolation: 'nearest_neighbor'},
       );

       expectArraysEqual(await output.data(), await expected.data());
       expectArraysEqual(output.shape, expected.shape);
     });

  it('smaller, resizeNearestNeighbor, same aspect ratio, alignCorners=true',
     async () => {
       const expectedShape: [number, number, number] = [2, 2, 4];
       const expected = tf.tensor3d(
           [
             [
               [200, 201, 202, 255],
               [170, 171, 172, 255],
             ],
             [
               [80, 81, 82, 255],
               [50, 51, 52, 255],
             ]
           ],
           expectedShape, 'int32');

       const output = fromTexture(
           gl,
           texture,
           {
             height: inShape[0],
             width: inShape[1],
             depth: inShape[2],
           },
           {
             height: expectedShape[0],
             width: expectedShape[1],
             depth: expectedShape[2],
           },
           {alignCorners: true, interpolation: 'nearest_neighbor'},
       );

       expectArraysEqual(await output.data(), await expected.data());
       expectArraysEqual(output.shape, expected.shape);
     });

  it('smaller, resizeNearestNeighbor, wider, alignCorners=false', async () => {
    const expectedShape: [number, number, number] = [2, 3, 4];
    const expected = tf.tensor3d(
        [
          [
            [200, 201, 202, 255],
            [190, 191, 192, 255],
            [180, 181, 182, 255],
          ],
          [
            [120, 121, 122, 255],
            [110, 111, 112, 255],
            [100, 101, 102, 255],
          ]
        ],
        expectedShape, 'int32');

    const output = fromTexture(
        gl,
        texture,
        {
          height: inShape[0],
          width: inShape[1],
          depth: inShape[2],
        },
        {
          height: expectedShape[0],
          width: expectedShape[1],
          depth: expectedShape[2],
        },
        {alignCorners: false, interpolation: 'nearest_neighbor'},
    );

    expectArraysEqual(await output.data(), await expected.data());
    expectArraysEqual(output.shape, expected.shape);
  });

  it('smaller, resizeNearestNeighbor, wider, alignCorners=true', async () => {
    const expectedShape: [number, number, number] = [2, 3, 4];
    const expected = tf.tensor3d(
        [
          [
            [200, 201, 202, 255],
            [180, 181, 182, 255],
            [170, 171, 172, 255],
          ],

          [
            [80, 81, 82, 255],
            [60, 61, 62, 255],
            [50, 51, 52, 255],
          ]
        ],
        expectedShape, 'int32');

    const output = fromTexture(
        gl,
        texture,
        {
          height: inShape[0],
          width: inShape[1],
          depth: inShape[2],
        },
        {
          height: expectedShape[0],
          width: expectedShape[1],
          depth: expectedShape[2],
        },
        {alignCorners: true, interpolation: 'nearest_neighbor'},
    );

    expectArraysEqual(await output.data(), await expected.data());
    expectArraysEqual(output.shape, expected.shape);
  });

  it('same size, should drop alpha channel', async () => {
    await detectGLCapabilities(gl);
    const expected = tf.tensor3d(
        [
          [
            [200, 201, 202],
            [190, 191, 192],
            [180, 181, 182],
            [170, 171, 172],
          ],
          [
            [160, 161, 162],
            [150, 151, 152],
            [140, 141, 142],
            [130, 131, 132],
          ],
          [
            [120, 121, 122],
            [110, 111, 112],
            [100, 101, 102],
            [90, 91, 92],
          ],
          [
            [80, 81, 82],
            [70, 71, 72],
            [60, 61, 62],
            [50, 51, 52],
          ]
        ],
        [inShape[0], inShape[1], 3], 'int32');

    const output = fromTexture(
        gl,
        texture,
        {
          height: inShape[0],
          width: inShape[1],
          depth: inShape[2],
        },
        {
          height: inShape[0],
          width: inShape[1],
          depth: 3,
        },
        {
          alignCorners: true,
          interpolation: 'nearest_neighbor',
        },
    );
    expectArraysEqual(await output.data(), await expected.data());
    expectArraysEqual(output.shape, expected.shape);
  });
});

describeWithFlags('fromTexture:bilinear', RN_ENVS, () => {
  let texture: WebGLTexture;
  let input: tf.Tensor3D;
  const inShape: [number, number, number] = [4, 4, 4];

  beforeAll(async () => {
    if (gl == null) {
      gl = await createGLContext();
    }

    input = tf.tensor3d(
        [
          [
            [200, 201, 202, 255],
            [190, 191, 192, 255],
            [180, 181, 182, 255],
            [170, 171, 172, 255],
          ],
          [
            [160, 161, 162, 255],
            [150, 151, 152, 255],
            [140, 141, 142, 255],
            [130, 131, 132, 255],
          ],
          [
            [120, 121, 122, 255],
            [110, 111, 112, 255],
            [100, 101, 102, 255],
            [90, 91, 92, 255],
          ],
          [
            [80, 81, 82, 255],
            [70, 71, 72, 255],
            [60, 61, 62, 255],
            [50, 51, 52, 255],
          ]
        ],
        inShape, 'int32');
  });

  afterAll(() => {
    tf.dispose(input);
  });

  beforeEach(async () => {
    texture = await toTexture(gl, input);
  });

  it('same size alignCorners=false', async () => {
    const output = fromTexture(
        gl,
        texture,
        {
          height: inShape[0],
          width: inShape[1],
          depth: inShape[2],
        },
        {
          height: inShape[0],
          width: inShape[1],
          depth: inShape[2],
        },
        {
          alignCorners: false,
          interpolation: 'bilinear',
        },
    );

    expectArraysEqual(await output.data(), await input.data());
    expectArraysEqual(output.shape, input.shape);
  });

  it('same size, alignCorners=true', async () => {
    const output = fromTexture(
        gl,
        texture,
        {
          height: inShape[0],
          width: inShape[1],
          depth: inShape[2],
        },
        {
          height: inShape[0],
          width: inShape[1],
          depth: inShape[2],
        },
        {
          alignCorners: true,
          interpolation: 'bilinear',
        },
    );

    expectArraysEqual(await output.data(), await input.data());
    expectArraysEqual(output.shape, input.shape);
  });

  it('smaller, same aspect ratio, alignCorners=false', async () => {
    const expectedShape: [number, number, number] = [2, 2, 4];
    const expected = tf.tensor3d(
        [
          [
            [200, 201, 202, 255],
            [180, 181, 182, 255],
          ],
          [
            [120, 121, 122, 255],
            [100, 101, 102, 255],
          ]
        ],
        expectedShape, 'int32');

    const output = fromTexture(
        gl,
        texture,
        {
          height: inShape[0],
          width: inShape[1],
          depth: inShape[2],
        },
        {
          height: expectedShape[0],
          width: expectedShape[1],
          depth: expectedShape[2],
        },
        {alignCorners: false, interpolation: 'bilinear'},
    );

    expectArraysEqual(await output.data(), await expected.data());
    expectArraysEqual(output.shape, expected.shape);
  });

  it('smaller, same aspect ratio, alignCorners=true', async () => {
    const expectedShape: [number, number, number] = [2, 2, 4];
    const expected = tf.tensor3d(
        [
          [
            [200, 201, 202, 255],
            [170, 171, 172, 255],
          ],
          [
            [80, 81, 82, 255],
            [50, 51, 52, 255],
          ]
        ],
        expectedShape, 'int32');

    const output = fromTexture(
        gl,
        texture,
        {
          height: inShape[0],
          width: inShape[1],
          depth: inShape[2],
        },
        {
          height: expectedShape[0],
          width: expectedShape[1],
          depth: expectedShape[2],
        },
        {alignCorners: true, interpolation: 'bilinear'},
    );

    expectArraysEqual(await output.data(), await expected.data());
    expectArraysEqual(output.shape, expected.shape);
  });

  it('smaller, wider, alignCorners=false', async () => {
    const expectedShape: [number, number, number] = [2, 3, 4];
    const expected = tf.tensor3d(
        [
          [
            [200, 201, 202, 255],
            [187, 188, 189, 255],
            [173, 174, 175, 255],
          ],
          [
            [120, 121, 122, 255],
            [107, 108, 109, 255],
            [93, 94, 95, 255],
          ]
        ],
        expectedShape, 'int32');

    const output = fromTexture(
        gl,
        texture,
        {
          height: inShape[0],
          width: inShape[1],
          depth: inShape[2],
        },
        {
          height: expectedShape[0],
          width: expectedShape[1],
          depth: expectedShape[2],
        },
        {alignCorners: false, interpolation: 'bilinear'},
    );

    expectArraysEqual(await output.data(), await expected.data());
    expectArraysEqual(output.shape, expected.shape);
  });

  it('smaller, wider, alignCorners=true', async () => {
    const expectedShape: [number, number, number] = [2, 3, 4];
    const expected = tf.tensor3d(
        [
          [
            [200, 201, 202, 255],
            [185, 186, 187, 255],
            [170, 171, 172, 255],
          ],
          [
            [80, 81, 82, 255],
            [65, 66, 67, 255],
            [50, 51, 52, 255],
          ]
        ],
        expectedShape, 'int32');

    const output = fromTexture(
        gl,
        texture,
        {
          height: inShape[0],
          width: inShape[1],
          depth: inShape[2],
        },
        {
          height: expectedShape[0],
          width: expectedShape[1],
          depth: expectedShape[2],
        },
        {alignCorners: true, interpolation: 'bilinear'},
    );

    expectArraysEqual(await output.data(), await expected.data());
    expectArraysEqual(output.shape, expected.shape);
  });

  it('same size, should drop alpha channel', async () => {
    await detectGLCapabilities(gl);
    const expected = tf.tensor3d(
        [
          [
            [200, 201, 202],
            [190, 191, 192],
            [180, 181, 182],
            [170, 171, 172],
          ],
          [
            [160, 161, 162],
            [150, 151, 152],
            [140, 141, 142],
            [130, 131, 132],
          ],
          [
            [120, 121, 122],
            [110, 111, 112],
            [100, 101, 102],
            [90, 91, 92],
          ],
          [
            [80, 81, 82],
            [70, 71, 72],
            [60, 61, 62],
            [50, 51, 52],
          ]
        ],
        [inShape[0], inShape[1], 3], 'int32');

    const output = fromTexture(
        gl,
        texture,
        {
          height: inShape[0],
          width: inShape[1],
          depth: inShape[2],
        },
        {
          height: inShape[0],
          width: inShape[1],
          depth: 3,
        },
        {
          alignCorners: true,
          interpolation: 'bilinear',
        },
    );
    expectArraysEqual(await output.data(), await expected.data());
    expectArraysEqual(output.shape, expected.shape);
  });
});
