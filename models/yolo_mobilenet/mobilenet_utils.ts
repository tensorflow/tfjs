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

export class BoundingBox {
  public x: number;
  public y: number;
  public w: number;
  public h: number;
  public c: number;
  public probs: Float32Array;

  private maxProb = -1;
  private maxIndx = -1;

  public static LABELS = ['raccoon'];
  public static COLORS = ['rgb(43,206,72)'];

  constructor(
      x: number, y: number, w: number, h: number, conf: number,
      probs: Float32Array) {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.c = conf;

    this.probs = probs;
  }

  public getMaxProb(): number {
    if (this.maxProb === -1) {
      this.maxProb = this.probs.reduce((a, b) => Math.max(a, b));
    }

    return this.maxProb;
  }

  public getLabel(): string {
    if (this.maxIndx === -1) {
      this.maxIndx = this.probs.indexOf(this.getMaxProb());
    }

    return BoundingBox.LABELS[this.maxIndx];
  }

  public getColor(): string {
    if (this.maxIndx === -1) {
      this.maxIndx = this.probs.indexOf(this.getMaxProb());
    }

    return BoundingBox.COLORS[this.maxIndx];
  }

  public iou(box: BoundingBox): number {
    const intersection = this.intersect(box);
    const union = this.w * this.h + box.w * box.h - intersection;

    return intersection / union;
  }

  private intersect(box: BoundingBox): number {
    const width = this.overlap(
        [this.x - this.w / 2, this.x + this.w / 2],
        [box.x - box.w / 2, box.x + box.w / 2]);
    const height = this.overlap(
        [this.y - this.h / 2, this.y + this.h / 2],
        [box.y - box.h / 2, box.y + box.h / 2]);

    return width * height;
  }

  private overlap(intervalA: [number, number], intervalB: [number, number]):
      number {
    const [x1, x2] = intervalA;
    const [x3, x4] = intervalB;

    if (x3 < x1) {
      if (x4 < x1) {
        return 0;
      } else {
        return Math.min(x2, x4) - x1;
      }
    } else {
      if (x2 < x3) {
        return 0;
      } else {
        return Math.min(x2, x4) - x3;
      }
    }
  }
}
