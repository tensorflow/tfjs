/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {fingerPrint64, hexToLong} from './hash_util';
import {ALL_ENVS, describeWithFlags} from './jasmine_util';

/**
 * The ACMRandom generator is for situations where extreme statistical quality
 * is not important. ACMRandom is useful for testing since it is seeded allowing
 * for reproducible results as well as low overhead so using it will
 * not affect test speed.
 */
class ACMRandom {
  static readonly MAX_INT32 = 2147483647;
  private seed: number;

  constructor(seed: number) {
    seed = seed & 0x7fffffff;
    if (seed === 0 || seed === ACMRandom.MAX_INT32) {
      seed = 1;
    }
    this.seed = seed;
  }

  private next() {
    const A = hexToLong('41A7');  // bits 14, 8, 7, 5, 2, 1, 0
    const MAX_INT32 = ACMRandom.MAX_INT32;
    // We are computing
    //       seed = (seed * A) % MAX_INT32,    where MAX_INT32 = 2^31-1
    //
    // seed must not be zero or MAX_INT32, or else all subsequent computed
    // values will be zero or MAX_INT32 respectively.  For all other values,
    // seed will end up cycling through every number in [1,MAX_INT32-1]
    const product = A.mul(this.seed);

    // Compute (product % MAX_INT32) using the fact that
    // ((x << 31) % MAX_INT32) == x.
    this.seed = product.shru(31).add(product.and(MAX_INT32)).getLowBits();
    // The first reduction may overflow by 1 bit, so we may need to repeat.
    // mod == MAX_INT32 is not possible; using > allows for the faster
    // sign-bit-based test.
    if (this.seed > MAX_INT32) {
      this.seed -= MAX_INT32;
    }
    return this.seed;
  }

  public rand8() {
    return (this.next() >> 1) & 0x000000ff;
  }
}

describeWithFlags('hash_util', ALL_ENVS, () => {
  it('check incremental hashes', () => {
    const buf = new Uint8Array(1000);
    const r = new ACMRandom(10);
    for (let i = 0; i < buf.length; ++i) {
      buf[i] = r.rand8();
    }

    const expectIteration = (length: number, expectedHash: string) =>
        expect(fingerPrint64(buf, length)).toEqual(hexToLong(expectedHash));

    expectIteration(0, '9ae16a3b2f90404f');
    expectIteration(1, '49d8a5e3fa93c327');
    expectIteration(2, 'fd259abb0ff2bf12');
    expectIteration(3, '781f1c6437096ac2');
    expectIteration(4, '0f1369d6c0b45716');
    expectIteration(5, '02d8cec6394de09a');
    expectIteration(7, '1faf6c6d43626c48');
    expectIteration(9, 'c93efd6dbe139be8');
    expectIteration(12, '65abbc967c87d515');
    expectIteration(16, '3f61450a03cff5af');
    expectIteration(20, '5d2fe297e45fed1a');
    expectIteration(26, 'eb665983aeb9ab94');
    expectIteration(33, '3a5f3890b124b4a3');
    expectIteration(42, 'f0c1cd66d7d9f246');
    expectIteration(53, 'cf7e3e4b1efeba6d');
    expectIteration(67, '0ced753b45740875');
    expectIteration(84, 'a585e0be01846ff4');
    expectIteration(105, 'fb6496deb356cdda');
    expectIteration(132, 'f2e4c5b6db6c154a');
    expectIteration(166, '4498451c3bca85a0');
    expectIteration(208, 'd604355fa4d0b14e');
    expectIteration(261, 'bb165e6b84ba9cdf');
    expectIteration(327, '9ebfab4519b1348c');
    expectIteration(409, '5921974ba2e9a5c2');
    expectIteration(512, 'a86a96e7a44282e3');
    expectIteration(640, 'dd731dfee500aa3c');
    expectIteration(800, 'a69f3400e6c98357');
    expectIteration(1000, '5c63b66443990bec');
  });

  // This is more thorough, but if something is wrong the output will be even
  // less illuminating, because it just checks one integer at the end.
  it('check many different strings', () => {
    const iters = 800;
    const s = new Uint8Array(4 * iters);
    let len = 0;
    let h = hexToLong('0');

    // Helper that replaces h with a hash of itself and return a
    // char that is also a hash of h.  Neither hash needs to be particularly
    // good.
    const remix = (): number => {
      h = h.xor(h.shru(41));
      h = h.mul(949921979);
      return 'a'.charCodeAt(0) + h.and(0xfffff).mod(26).getLowBits();
    };

    for (let i = 0; i < iters; i++) {
      h = h.xor(fingerPrint64(s, i));
      s[len++] = remix();
      h = h.xor(fingerPrint64(s, i * i % len));
      s[len++] = remix();
      h = h.xor(fingerPrint64(s, i * i * i % len));
      s[len++] = remix();
      h = h.xor(fingerPrint64(s, len));
      s[len++] = remix();
      const x0 = s[len - 1];
      const x1 = s[len - 2];
      const x2 = s[len - 3];
      const x3 = s[len >> 1];
      s[((x0 << 16) + (x1 << 8) + x2) % len] ^= x3;
      s[((x1 << 16) + (x2 << 8) + x3) % len] ^= i % 256;
    }

    expect(h).toEqual(hexToLong('7a1d67c50ec7e167'));
  });

  it('check string hash', () => {
    const fingerPrintHash = (hash: Long) => {
      const mul = hexToLong('9ddfea08eb382d69');
      let b = hash.mul(mul);
      b = b.xor(b.shru(44));
      b = b.mul(mul);
      b = b.xor(b.shru(41));
      b = b.mul(mul);
      return b;
    };

    const getString = (length: number) => Uint8Array.from(
        'x'.repeat(length).split('').map(char => char.charCodeAt(0)));

    expect(fingerPrintHash(fingerPrint64(getString(40))))
        .toEqual(hexToLong('2117170c4aebaffe'));
    expect(fingerPrintHash(fingerPrint64(getString(60))))
        .toEqual(hexToLong('e252963f3fd7a3af'));
    expect(fingerPrintHash(fingerPrint64(getString(70))))
        .toEqual(hexToLong('b0a8cf4a56c570fa'));
    expect(fingerPrintHash(fingerPrint64(getString(80))))
        .toEqual(hexToLong('d6ddaa49ddef5839'));
    expect(fingerPrintHash(fingerPrint64(getString(90))))
        .toEqual(hexToLong('168f3a694b4dce29'));
  });
});
