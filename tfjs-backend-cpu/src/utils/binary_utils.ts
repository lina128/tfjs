/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {backend_util, DataType, TypedArray, util} from '@tensorflow/tfjs-core';

export interface BroadcastedBinaryOpResult {
  values: TypedArray;
  shape: number[];
}

export interface BroadcastedBinaryComplexOpResult {
  realVals: TypedArray;
  imagVals: TypedArray;
  shape: number[];
}

export function broadcastedBinaryOp(
    aVals: TypedArray, bVals: TypedArray, aShape: number[], bShape: number[],
    dtype: DataType,
    op: (a: number, b: number) => number): BroadcastedBinaryOpResult {
  const newShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);

  const aBroadcastDims = backend_util.getBroadcastDims(aShape, newShape);
  const bBroadcastDims = backend_util.getBroadcastDims(bShape, newShape);

  const resVals =
      util.getArrayFromDType(dtype, util.sizeFromShape(newShape)) as TypedArray;

  if (aBroadcastDims.length + bBroadcastDims.length === 0) {
    for (let i = 0; i < resVals.length; ++i) {
      resVals[i] = op(aVals[i % aVals.length], bVals[i % bVals.length]);
    }
  } else {
    const aStrides = util.computeStrides(aShape);
    const bStrides = util.computeStrides(bShape);
    const outStrides = util.computeStrides(newShape);

    for (let i = 0; i < resVals.length; ++i) {
      const aIndex = convertBroadcastedIndexToOriginalIndex(
          i, aShape, aBroadcastDims, aStrides, newShape, outStrides);
      const bIndex = convertBroadcastedIndexToOriginalIndex(
          i, bShape, bBroadcastDims, bStrides, newShape, outStrides);

      resVals[i] = op(aVals[aIndex], bVals[bIndex]);
    }
  }

  return {values: resVals, shape: newShape};
}

export function broadcastedBinaryComplexOp(
    aRealVals: TypedArray, aImagVals: TypedArray, bRealVals: TypedArray,
    bImagVals: TypedArray, aShape: number[], bShape: number[],
    op: (aReal: number, aImag: number, bReal: number, bImag: number) => {
      real: number,
      imag: number
    }): BroadcastedBinaryComplexOpResult {
  const newShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);

  const aBroadcastDims = backend_util.getBroadcastDims(aShape, newShape);
  const bBroadcastDims = backend_util.getBroadcastDims(bShape, newShape);

  const realVals = util.getArrayFromDType(
                       'float32', util.sizeFromShape(newShape)) as TypedArray;

  const imagVals = util.getArrayFromDType(
                       'float32', util.sizeFromShape(newShape)) as TypedArray;

  if (aBroadcastDims.length + bBroadcastDims.length === 0) {
    for (let i = 0; i < realVals.length; i++) {
      const aIdx = i % aRealVals.length;
      const bIdx = i % bRealVals.length;

      const result = op(
          aRealVals[aIdx], aImagVals[aIdx], bRealVals[bIdx], bImagVals[bIdx]);

      realVals[i] = result.real;
      imagVals[i] = result.imag;
    }
  } else {
    const aStrides = util.computeStrides(aShape);
    const bStrides = util.computeStrides(bShape);
    const outStrides = util.computeStrides(newShape);

    for (let i = 0; i < realVals.length; i++) {
      const aIndex = convertBroadcastedIndexToOriginalIndex(
          i, aShape, aBroadcastDims, aStrides, newShape, outStrides);
      const bIndex = convertBroadcastedIndexToOriginalIndex(
          i, bShape, bBroadcastDims, bStrides, newShape, outStrides);

      const opResult =
          op(aRealVals[aIndex], aImagVals[aIndex], bRealVals[bIndex],
             bImagVals[bIndex]);

      realVals[i] = opResult.real;
      imagVals[i] = opResult.imag;
    }
  }

  return {realVals, imagVals, shape: newShape};
}

function convertBroadcastedIndexToOriginalIndex(
    broadCastedIndex: number, shape: number[], originalBroadcastDims: number[],
    strides: number[], outShape: number[], outStrides: number[]): number {
  const coord = convertIndexToCoord(broadCastedIndex, outShape, outStrides);
  const originalCoord = coord.slice(-shape.length);

  originalBroadcastDims.forEach(d => originalCoord[d] = 0);

  return convertCoordToIndex(originalCoord, shape, strides);
}

function convertIndexToCoord(
    index: number, shape: number[], strides: number[]): number[] {
  const rank = shape.length;

  if (rank === 0) {
    return [];
  } else if (rank === 1) {
    return [index];
  }
  const coord: number[] = new Array(rank);

  for (let i = 0; i < coord.length - 1; ++i) {
    coord[i] = Math.floor(index / strides[i]);
    index -= coord[i] * strides[i];
  }

  coord[coord.length - 1] = index;

  return coord;
}

function convertCoordToIndex(
    coord: number[], shape: number[], strides: number[]): number {
  const rank = shape.length;

  if (rank === 0) {
    return 0;
  } else if (rank === 1) {
    return coord[0];
  }

  let index = coord[coord.length - 1];

  for (let i = 0; i < coord.length - 1; ++i) {
    index += strides[i] * coord[i];
  }

  return index;
}
