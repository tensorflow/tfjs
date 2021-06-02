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
// Workaround for allowing cjs module to be included in bundle created by
// rollup.
import * as LongExports from 'long';
// tslint:disable-next-line
const Long: LongExports.LongConstructor =
    // tslint:disable-next-line
    (LongExports as any).default || LongExports;

export function hexToLong(hex: string): Long {
  return Long.fromString(hex, true, 16);
}

// Some primes between 2^63 and 2^64 for various uses.
// Hex 0xc3a5c85c97cb3127
const k0: Long = hexToLong('c3a5c85c97cb3127');
// Hex 0xb492b66fbe98f273
const k1: Long = hexToLong('b492b66fbe98f273');
// Hex 0x9ae16a3b2f90404f
const k2: Long = hexToLong('9ae16a3b2f90404f');

function shiftMix(val: Long): Long {
  return val.xor(val.shru(47));
}

function fetch(s: Uint8Array, offset: number, numBytes: number): Long {
  const bytes = s.slice(offset, offset + numBytes);
  return Long.fromBytes(Array.from(bytes), true, true);
}

function fetch64(s: Uint8Array, offset: number): Long {
  return fetch(s, offset, 8);
}

function fetch32(s: Uint8Array, offset: number): Long {
  return fetch(s, offset, 4);
}

function rotate64(val: Long, shift: number): Long {
  // Avoid shifting by 64: doing so yields an undefined result.
  return shift === 0 ? val : val.shru(shift).or(val.shl(64 - shift));
}

function hashLen16(u: Long, v: Long, mul = hexToLong('9ddfea08eb382d69')) {
  // Murmur-inspired hashing.
  let a = u.xor(v).mul(mul);
  a = a.xor(a.shru(47));
  let b = v.xor(a).mul(mul);
  b = b.xor(b.shru(47));
  b = b.mul(mul);
  return b;
}

// Return a 16-byte hash for 48 bytes.  Quick and dirty.
// Callers do best to use "random-looking" values for a and b.
function weakHashLen32WithSeeds(
    w: Long, x: Long, y: Long, z: Long, a: Long, b: Long) {
  a = a.add(w);
  b = rotate64(b.add(a).add(z), 21);
  const c = a;
  a = a.add(x);
  a = a.add(y);
  b = b.add(rotate64(a, 44));
  return [a.add(z), b.add(c)];
}

function weakHashLen32WithSeedsStr(
    s: Uint8Array, offset: number, a: Long, b: Long) {
  return weakHashLen32WithSeeds(
      fetch64(s, offset), fetch64(s, offset + 8), fetch64(s, offset + 16),
      fetch64(s, offset + 24), a, b);
}

function hashLen0to16(s: Uint8Array, len = s.length): Long {
  if (len >= 8) {
    const mul = k2.add(len * 2);
    const a = fetch64(s, 0).add(k2);
    const b = fetch64(s, len - 8);
    const c = rotate64(b, 37).mul(mul).add(a);
    const d = rotate64(a, 25).add(b).mul(mul);
    return hashLen16(c, d, mul);
  }
  if (len >= 4) {
    const mul = k2.add(len * 2);
    const a = fetch32(s, 0);
    return hashLen16(a.shl(3).add(len), fetch32(s, len - 4), mul);
  }
  if (len > 0) {
    const a = s[0];
    const b = s[len >> 1];
    const c = s[len - 1];
    const y = a + (b << 8);
    const z = len + (c << 2);
    return shiftMix(k2.mul(y).xor(k0.mul(z))).mul(k2);
  }
  return k2;
}

function hashLen17to32(s: Uint8Array, len = s.length): Long {
  const mul = k2.add(len * 2);
  const a = fetch64(s, 0).mul(k1);
  const b = fetch64(s, 8);
  const c = fetch64(s, len - 8).mul(mul);
  const d = fetch64(s, len - 16).mul(k2);
  return hashLen16(
      rotate64(a.add(b), 43).add(rotate64(c, 30)).add(d),
      a.add(rotate64(b.add(k2), 18)).add(c), mul);
}

function hashLen33to64(s: Uint8Array, len = s.length): Long {
  const mul = k2.add(len * 2);
  const a = fetch64(s, 0).mul(k2);
  const b = fetch64(s, 8);
  const c = fetch64(s, len - 8).mul(mul);
  const d = fetch64(s, len - 16).mul(k2);
  const y = rotate64(a.add(b), 43).add(rotate64(c, 30)).add(d);
  const z = hashLen16(y, a.add(rotate64(b.add(k2), 18)).add(c), mul);
  const e = fetch64(s, 16).mul(mul);
  const f = fetch64(s, 24);
  const g = y.add(fetch64(s, len - 32)).mul(mul);
  const h = z.add(fetch64(s, len - 24)).mul(mul);
  return hashLen16(
      rotate64(e.add(f), 43).add(rotate64(g, 30)).add(h),
      e.add(rotate64(f.add(a), 18)).add(g), mul);
}

export function fingerPrint64(s: Uint8Array, len = s.length): Long {
  const seed: Long = Long.fromNumber(81, true);
  if (len <= 32) {
    if (len <= 16) {
      return hashLen0to16(s, len);
    } else {
      return hashLen17to32(s, len);
    }
  } else if (len <= 64) {
    return hashLen33to64(s, len);
  }

  // For strings over 64 bytes we loop.  Internal state consists of
  // 56 bytes: v, w, x, y, and z.
  let x = seed;
  let y = seed.mul(k1).add(113);

  let z = shiftMix(y.mul(k2).add(113)).mul(k2);
  let v = [Long.UZERO, Long.UZERO];
  let w = [Long.UZERO, Long.UZERO];
  x = x.mul(k2).add(fetch64(s, 0));

  let offset = 0;
  // Set end so that after the loop we have 1 to 64 bytes left to process.
  const end = ((len - 1) >> 6) * 64;
  const last64 = end + ((len - 1) & 63) - 63;

  do {
    x = rotate64(x.add(y).add(v[0]).add(fetch64(s, offset + 8)), 37).mul(k1);
    y = rotate64(y.add(v[1]).add(fetch64(s, offset + 48)), 42).mul(k1);
    x = x.xor(w[1]);
    y = y.add(v[0]).add(fetch64(s, offset + 40));
    z = rotate64(z.add(w[0]), 33).mul(k1);
    v = weakHashLen32WithSeedsStr(s, offset, v[1].mul(k1), x.add(w[0]));
    w = weakHashLen32WithSeedsStr(
        s, offset + 32, z.add(w[1]), y.add(fetch64(s, offset + 16)));

    [z, x] = [x, z];
    offset += 64;
  } while (offset !== end);
  const mul = k1.add(z.and(0xff).shl(1));
  // Point to the last 64 bytes of input.
  offset = last64;

  w[0] = w[0].add((len - 1) & 63);
  v[0] = v[0].add(w[0]);
  w[0] = w[0].add(v[0]);

  x = rotate64(x.add(y).add(v[0]).add(fetch64(s, offset + 8)), 37).mul(mul);
  y = rotate64(y.add(v[1]).add(fetch64(s, offset + 48)), 42).mul(mul);
  x = x.xor(w[1].mul(9));
  y = y.add(v[0].mul(9).add(fetch64(s, offset + 40)));
  z = rotate64(z.add(w[0]), 33).mul(mul);
  v = weakHashLen32WithSeedsStr(s, offset, v[1].mul(mul), x.add(w[0]));
  w = weakHashLen32WithSeedsStr(
      s, offset + 32, z.add(w[1]), y.add(fetch64(s, offset + 16)));

  [z, x] = [x, z];

  return hashLen16(
      hashLen16(v[0], w[0], mul).add(shiftMix(y).mul(k0)).add(z),
      hashLen16(v[1], w[1], mul).add(x), mul);
}
