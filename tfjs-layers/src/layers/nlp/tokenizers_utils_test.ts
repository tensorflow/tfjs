/**
 * @license
 * Copyright 2023 Google LLC.
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

import { tensor, test_util } from '@tensorflow/tfjs-core';

import { BytePairTokenizerCache, SPLIT_PATTERN_1, bytesToUnicode,
  createAltsForUnsplittableTokens, createStaticHashtable, regexSplit,
  removeStringsFromInputs, splitStringsForBpe } from './tokenizers_utils';
import { expectTensorsClose } from '../../utils/test_utils';

describe('bytesToUnicode', () => {
  it('returns correct output', () => {
    const [bytesList, charsList] = bytesToUnicode();
    const expectedChars = ['!', '"', '#', '$', '%', '&', '\'', '(', ')', '*',
      '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
      ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
      'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
      'X','Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f',
      'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
      'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '¡', '¢', '£', '¤', '¥', '¦',
      '§', '¨', '©', 'ª', '«', '¬', '®', '¯', '°', '±', '²', '³', '´', 'µ', '¶',
      '·', '¸', '¹', 'º', '»', '¼', '½', '¾', '¿', 'À', 'Á', 'Â', 'Ã', 'Ä', 'Å',
      'Æ', 'Ç', 'È', 'É', 'Ê', 'Ë', 'Ì', 'Í', 'Î', 'Ï', 'Ð', 'Ñ', 'Ò', 'Ó', 'Ô',
      'Õ', 'Ö', '×', 'Ø', 'Ù', 'Ú', 'Û', 'Ü', 'Ý', 'Þ', 'ß', 'à', 'á', 'â', 'ã',
      'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ð', 'ñ', 'ò',
      'ó', 'ô', 'õ', 'ö', '÷', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ÿ', 'Ā', 'ā',
      'Ă', 'ă', 'Ą', 'ą', 'Ć', 'ć', 'Ĉ', 'ĉ', 'Ċ', 'ċ', 'Č', 'č', 'Ď', 'ď', 'Đ',
      'đ', 'Ē', 'ē', 'Ĕ', 'ĕ', 'Ė', 'ė', 'Ę', 'ę', 'Ě', 'ě', 'Ĝ', 'ĝ', 'Ğ', 'ğ',
      'Ġ', 'ġ', 'Ģ', 'ģ', 'Ĥ', 'ĥ', 'Ħ', 'ħ', 'Ĩ', 'ĩ', 'Ī', 'ī', 'Ĭ', 'ĭ', 'Į',
      'į', 'İ', 'ı', 'Ĳ', 'ĳ', 'Ĵ', 'ĵ', 'Ķ', 'ķ', 'ĸ', 'Ĺ', 'ĺ', 'Ļ', 'ļ', 'Ľ',
      'ľ', 'Ŀ', 'ŀ', 'Ł', 'ł', 'Ń'];
    const expectedBytes = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
      46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
      64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
      82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
      100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
      115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 161, 162, 163,
      164, 165, 166, 167, 168, 169, 170, 171, 172, 174, 175, 176, 177, 178, 179,
      180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
      195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
      210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
      225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
      240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
      255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
      20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 127, 128, 129, 130,
      131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
      146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
      173];

    test_util.expectArraysClose(bytesList, expectedBytes);
    test_util.expectArraysClose(charsList, expectedChars);
  });
});

describe('createStaticHashtable', () => {
  it('creates StaticHashTable<number, string> correctly', () => {
    const [bytesList, charsList] = bytesToUnicode();
    const byte2Unicode = createStaticHashtable(
      Array.from(bytesList), charsList, '');

    expect(byte2Unicode.get(33)).toBe('!');
    expect(byte2Unicode.get(-1)).toBe('');

    expectTensorsClose(
      (byte2Unicode.lookup([tensor([33, 133])]))[0],
      tensor(['!', '\x85']));
  });

  it('creates StaticHashTable<string, number> correctly', () => {
    const [bytesList, charsList] = bytesToUnicode();
    const unicode2Byte = createStaticHashtable(
      charsList, Array.from(bytesList), -1);

    expect(unicode2Byte.get('Û')).toBe(219);
    expect(unicode2Byte.get('😁')).toBe(-1);

    expectTensorsClose(
      (unicode2Byte.lookup([tensor(['!', '{'])]))[0],
      tensor([33, 123]));
  });
});

describe('BytePairTokenizerCache', () => {
  let cache: BytePairTokenizerCache;

  beforeEach(() => {
    cache = new BytePairTokenizerCache();
  });

  it('inserts strings and retrieves correctly', () => {
    cache.insert(['butterfly', 'dragonfly'], ['but ter fly', 'dragon fly']);

    test_util.expectArraysEqual(cache.lookup(['butterfly']), ['but ter fly']);
  });

  it('inserts tensors and retrieves correctly', () => {
    cache.insert(
      tensor(['butterfly', 'dragonfly']), ['but ter fly', 'dragon fly']);

    test_util.expectArraysEqual(
      cache.lookup(tensor(['dragonfly'])), ['dragon fly']);
  });
});

describe('removeStringsFromInputs', () => {
  it ('removes nothing successfully', () => {
    const inputs = [tensor(['butterfly']), tensor(['butter'])];
    const stringToRemove = '६';

    const result = removeStringsFromInputs(inputs, stringToRemove);

    expect(result.length).toBe(2);
    expectTensorsClose(result[0], tensor(['butterfly']));
    expectTensorsClose(result[1], tensor(['butter']));
  });

  it ('removes strings successfully', () => {
    const inputs = [tensor(['butterfly']), tensor(['butter'])];
    const stringToRemove = 'butter';

    const result = removeStringsFromInputs(inputs, stringToRemove);

    expect(result.length).toBe(2);
    expectTensorsClose(result[0], tensor(['butterfly']));
    expectTensorsClose(result[1], tensor([]));
  });
});

describe('createAltsForUnsplittableTokens', () => {
  it ('creates alts with no matching regex pattern', () => {
    const unsplittableTokens = ['s', 'p'];

    const result = createAltsForUnsplittableTokens(unsplittableTokens);

    expect(result.length).toBe(2);
    test_util.expectArraysEqual(result, ['ĵs', 'ĵp']);
  });

  it ('creates alts with matching regex pattern', () => {
    const unsplittableTokens = [' s', 'p'];

    const result = createAltsForUnsplittableTokens(unsplittableTokens);

    expect(result.length).toBe(2);
    test_util.expectArraysEqual(result, ['ĵs', 'ĵp']);
  });

  it ('regex works correctly', () => {
    const unsplittableTokens = ['😊,_五$ñü]aA5{\'\n~`'];

    const result = createAltsForUnsplittableTokens(unsplittableTokens);

    expect(result.length).toBe(1);
    test_util.expectArraysEqual(result, ['ĵ五ñüaA5']);
  });
});

describe('regexSplit', () => {
  it ('splits with regex and string', () => {
    const strResult = regexSplit(['hello there'], /\s/g);
    const regexResult = regexSplit(['hello there'], ' ');
    const expected = [['hello', 'there']];

    test_util.expectArraysEqual(strResult, expected);
    test_util.expectArraysEqual(regexResult, expected);
  });

  it ('keeps string delimiter', () => {
    test_util.expectArraysEqual(regexSplit(['sp'], 's', true), [['s', 'p']]);
    test_util.expectArraysEqual(
      regexSplit(['\xc4\xb4s', 'p'], 'p', true), [['Ä´s'], ['p']] );
  });

  it('splits regex delimiter', () => {
    const result = regexSplit(['ĵs', 'ĵp'], SPLIT_PATTERN_1, true);

    test_util.expectArraysEqual(result, [['ĵs'], ['ĵp']]);
  });

  it('works with periods', () => {
    const result = regexSplit(
      ['brown.', 'black.'], SPLIT_PATTERN_1, true);

    test_util.expectArraysEqual(result, [['brown', '.'], ['black', '.']]);
  });
});

describe('splitStringsForBpe', () => {
  it ('splits with unsplittable tokens', () => {
    const inputs = tensor(['sp']);
    const unsplittableTokens = ['s', 'p'];

    const result = splitStringsForBpe(inputs, unsplittableTokens);

    expect(result.length).toBe(1);
    expectTensorsClose(result[0], tensor(['s', 'p']));
  });

  it ('splits with multicharacter unsplittable tokens', () => {
    const inputs = tensor(['butterfly<|endOfText|>']);
    const unsplittableTokens = ['<|endOfText|>'];

    const result = splitStringsForBpe(inputs, unsplittableTokens);

    expect(result.length).toBe(1);
    expectTensorsClose(result[0], tensor(['butterfly', '<|endOfText|>']));
  });

  it ('splits with no unsplittable tokens', () => {
    const inputs = tensor(['brown.', 'black.']);

    const result = splitStringsForBpe(inputs);

    expect(result.length).toBe(2);
    expectTensorsClose(result[0], tensor(['brown', '.']));
    expectTensorsClose(result[1], tensor(['black', '.']));
  });
});
