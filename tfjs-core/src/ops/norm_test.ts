/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {ENGINE} from '../engine';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose/*, expectArraysEqual*/} from '../test_util';
import {norm} from './norm';

describeWithFlags('norm', ALL_ENVS, () => {
  it('computes Euclidean norm for 1D tensors correctly', async () => {
    const vectors = [
      [3.7],
      [4.2, 80.085, 13.37],
      [0],
      [0,0]
    ];
    Object.freeze(vectors);

    for( const vec of vectors )
    {
      const ref = Math.hypot(...vec);
      Object.freeze(ref);

      for( const p of ['euclidean', 2] as Array<'euclidean' | 2> ) {
      for( const axis of [undefined, 0, [0]] ) {
        for( const keepDims of [undefined,false] ) {
          const n = norm(vec, p, axis, keepDims);
          expectArraysClose(await n.array(), ref);
        }

        const n = norm(vec, p, axis, true);
        expectArraysClose(await n.array(), [ref]);
      }}
    }
  });

  it('computes Euclidean norm for 2D tensors correctly', async () => {
    const matrices = [
      [[0]],

      [[ 4.2 ],
       [13.37]],

      [[0.0, 0.0],
       [0.0, 4.2]]
    ];

    for( const p of ['euclidean', 2] as Array<'euclidean' | 2> ) {
    for( const mat of matrices ) {
      {
        const ref = mat.map( row => Math.hypot(...row) );
        Object.freeze(ref);

        for( const axis of [1,[-1]] ) {
          for( const keepDims of [undefined,false] ) {
            const n = norm(mat, p, axis, keepDims);
            expectArraysClose(await n.array(), ref);
          }

          const n = norm(mat, p, axis, true);
          expectArraysClose(
            await n.array(),
            ref.map(x => [x])
          );
        }
      }

      {
        const ref = mat.reduce(
          (norm,row) => norm.map((x,i) => Math.hypot(x,row[i])),
          /*init=*/mat[0].map(() => 0)
        );
        Object.freeze(ref);

        for( const axis of [-2,[0]] ) {
          for( const keepDims of [undefined,false] ) {
            const n = norm(mat, p, axis, keepDims);
            expectArraysClose(await n.array(), ref);
          }

          const n = norm(mat, p, axis, true);
          expectArraysClose(await n.array(), [ref]);
        }
      }
    }}
  });

  it('computes Euclidean norm for 3D tensors correctly', async () => {
    const arrays = [
      [[[0.0, 7.7, 0.0],
        [3.8, 1.3, 3.2]],
       [[4.7, 6.1, 1.6],
        [9.9, 2.5, 8.3]]],

      [[[0.0, 7.7],
        [0.0, 0.0],
        [0.0, 3.2]],
       [[4.7, 6.1],
        [0.0, 0.0],
        [2.5, 8.3]]]
    ];

    for( const p of ['euclidean', 2] as Array<'euclidean' | 2> ) {
    for( const arr of arrays ) {
      {
        const ref = arr.map( mat =>
                    mat.map( row => Math.hypot(...row) ));
        Object.freeze(ref);

        for( const axis of [2,[-1]] ) {
          for( const keepDims of [undefined,false] ) {
            const n = norm(arr, p,  axis, keepDims);
            expectArraysClose(await n.array(), ref);
          }

          const n = norm(arr, p, axis, true);
          expectArraysClose(
            await n.array(),
            arr.map( mat =>
            mat.map( row => [Math.hypot(...row)] ))
          );
        }
      }

      {
        const ref = arr.map( mat => mat.reduce(
          (norm,row) => norm.map((x,i) => Math.hypot(x,row[i])),
          /*init=*/mat[0].map(() => 0)
        ));
        Object.freeze(ref);

        for( const axis of [-2,[1]] )
        {
          for( const keepDims of [undefined,false] ) {
            const n = norm(arr, p,  axis, keepDims);
            expectArraysClose(await n.array(), ref);
          }

          const n = norm(arr, p, axis, true);
          expectArraysClose( await n.array(), ref.map(x => [x]) );
        }
      }

      {
        const ref = arr.reduce(
          (nrm,mat) => nrm.map( (row,i) =>
                       row.map( (nij,j) => Math.hypot(nij,mat[i][j]) )),
          /*init=*/arr[0].map( row => row.map(() => 0) )
        );
        Object.freeze(ref);

        for( const axis of [0,[-3]] )
        {
          for( const keepDims of [undefined,false] ) {
            const n = norm(arr, p,  axis, keepDims);
            expectArraysClose(await n.array(), ref);
          }

          const n = norm(arr, p, axis, true);
          expectArraysClose( await n.array(), [ref]);
        }
      }
    }}
  });

  it('computes Frobenius norm for 2D tensors correctly', async () => {
    const matrices = [
      [[0]],

      [[ 4.2 ],
       [13.37]],

      [[0.0, 0.0],
       [0.0, 4.2]]
    ];

    for( const p of ['euclidean', 'fro'] as Array<'euclidean' | 'fro'> ) {
    for( const mat of matrices ) {
      const axeBodySpray = [[ 0, 1],
                            [-2, 1],
                            [-1, 0],
                            [-1,-2]];
      if( p === 'euclidean' ) {
        axeBodySpray.push(null);
      }
      for( const axes of axeBodySpray ) {
        const ref = Math.hypot(...Array.from(
          function*(){
            for( const row of mat ) {
              yield* row;
            }
          }()
        ));

        for( const keepDims of [undefined,false] ) {
          const n = norm(mat, p, axes, keepDims);
          expectArraysClose(await n.array(), ref);
        }

        const n = norm(mat, p, axes, true);
        expectArraysClose(await n.array(), [[ref]]);
      }
    }}
  });

  it('computes Frobenius norm for 3D tensors correctly', async () => {
    const arrays = [
      [[[0.0, 7.7, 0.0],
        [3.8, 1.3, 3.2]],
       [[4.7, 6.1, 1.6],
        [9.9, 2.5, 8.3]]],

      [[[0.0, 7.7],
        [0.0, 0.0],
        [0.0, 3.2]],
       [[4.7, 6.1],
        [0.0, 0.0],
        [2.5, 8.3]]]
    ];

    for( const p of ['euclidean', 'fro'] as Array<'euclidean' | 'fro'> ) {
    for( const arr of arrays ) {
      for( const axes of [[ 1, 2],
                          [-1,-2]] ) {
        const ref = arr.map( mat => Math.hypot(...Array.from(
          function*(){
            for( const row of mat ) {
              yield *row;
            }
          }()
        )) );

        for( const keepDims of [undefined,false] ) {
          const n = norm(arr, p,  axes, keepDims);
          expectArraysClose(await n.array(), ref);
        }

        const n = norm(arr, p, axes, true);
        expectArraysClose(
          await n.array(),
          ref.map(x => [[x]])
        );
      }

      for( const axes of [[ 2, 0],
                          [-1,-3]] )
      {
        const ref = Array.from(arr[0], (_,j) => Math.hypot(...Array.from(
          function*(){
            for( let i=arr      .length; i-- > 0; ) {
            for( let k=arr[0][0].length; k-- > 0; ) {
              yield arr[i][j][k];
            }}
          }()
        )) );

        for( const keepDims of [undefined,false] ) {
          const n = norm(arr, p,  axes, keepDims);
          expectArraysClose(await n.array(), ref);
        }

        const n = norm(arr, p, axes, true);
        expectArraysClose(
          await n.array(),
          [ref.map(x => [x])]
        );
      }

      for( const axes of [[ 0,-2],
                          [ 1,-3]] )
      {
        const ref = Array.from(arr[0][0], (_,k) => Math.hypot(...Array.from(
          function*(){
            for( let i=arr   .length; i-- > 0; ) {
            for( let j=arr[0].length; j-- > 0; ) {
              yield arr[i][j][k];
            }}
          }()
        )) );

        for( const keepDims of [undefined,false] ) {
          const n = norm(arr, p,  axes, keepDims);
          expectArraysClose(await n.array(), ref);
        }

        const n = norm(arr, p, axes, true);
        expectArraysClose(await n.array(), [[ref]]);
      }
    }}
  });

  it('computes Euclidean norm underflow-safely', async () => {
    const small = (() => {
      const  floatBits = ENGINE.backend.floatPrecision();
      switch(floatBits) {
        case 32: return 1e-30;
        case 16: return 1e-4;
        default:
          throw new Error(
            'Test not implemented for ' +
            `ENGINE.backend.floatPrecision()=${floatBits}.`
          );
      }
    })();

    const tolerance = small / 100;

    const vectors = [
      [        small],
      [      0,small],
      [  small,    0],
      [  small,small],
      [0,    0,small],
      [0,small,    0],
      [0,small,small]
    ];

    for( const p of ['euclidean', 2] as Array<'euclidean' | 2> ) {
    for( const vec  of vectors ) {
    for( const axis of [undefined, 0, [0]] ) {
      const  ref = Math.hypot(...vec);
      expect(ref).toBeGreaterThan(0);

      for( const keepDims of [undefined,false] ) {
        const n = norm(vec, p, axis, keepDims);
        expectArraysClose(await n.array(), ref, tolerance);
      }

      const n = norm(vec, p, axis, true);
      expectArraysClose(await n.array(), [ref], tolerance);
    }}}
  });

  it('computes Frobenius norm for 2D tensors correctly', async () => {
    const small = (() => {
      const  floatBits = ENGINE.backend.floatPrecision();
      switch(floatBits) {
        case 32: return 1e-30;
        case 16: return 1e-4;
        default:
          throw new Error(
            'Test not implemented for ' +
            `ENGINE.backend.floatPrecision()=${floatBits}.`
          );
      }
    })();

    const tolerance = small / 100;

    const matrices = [
      [[small]],

      [[small],
       [small]],

      [[small,  0.0,small],
       [  0.0,small,small]]
    ];

    for( const p of ['euclidean', 'fro'] as Array<'euclidean' | 'fro'> ) {
    for( const mat of matrices ) {
      const axeBodySpray = [[ 0, 1],
                            [-2, 1],
                            [-1, 0],
                            [-1,-2]];
      if( p === 'euclidean' ) {
        axeBodySpray.push(null);
      }
      for( const axes of axeBodySpray ) {
        const ref = Math.hypot(...Array.from(
          function*(){
            for( const row of mat ) {
              yield* row;
            }
          }()
        ));
        expect(ref).toBeGreaterThan(0);

        for( const keepDims of [undefined,false] ) {
          const n = norm(mat, p, axes, keepDims);
          expectArraysClose(await n.array(), ref, tolerance);
        }

        const n = norm(mat, p, axes, true);
        expectArraysClose(await n.array(), [[ref]], tolerance);
      }
    }}
  });
});
