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

import {backend_util, DepthwiseConv2dNative, DepthwiseConv2dNativeAttrs, DepthwiseConv2dNativeInputs, KernelConfig, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export const depthwiseConv2dNativeConfig: KernelConfig = {
  kernelName: DepthwiseConv2dNative,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {x, filter} = inputs as DepthwiseConv2dNativeInputs;
    const {strides, pad, dilations, dimRoundingMode} =
        attrs as {} as DepthwiseConv2dNativeAttrs;
    const cpuBackend = backend as MathBackendCPU;

    assertNotComplex([x, filter], 'depthwiseConv2D');

    let $dilations = dilations;
    if (dilations == null) {
      $dilations = [1, 1];
    }

    util.assert(
        backend_util.eitherStridesOrDilationsAreOne(strides, $dilations),
        () => 'Error in depthwiseConv2d: Either strides or dilations must be ' +
            `1. Got strides ${strides} and dilations '${dilations}'`);

    const convInfo = backend_util.computeConv2DInfo(
        x.shape as [number, number, number, number],
        filter.shape as [number, number, number, number], strides, $dilations,
        pad, dimRoundingMode, true /* depthwise */);

    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const padLeft = convInfo.padInfo.left;
    const padTop = convInfo.padInfo.top;
    const chMul = convInfo.outChannels / convInfo.inChannels;

    const yVals = util.getArrayFromDType(
                      x.dtype as 'float32',
                      util.sizeFromShape(convInfo.outShape)) as TypedArray;

    const xStrides = util.computeStrides(x.shape);
    const filterStrides = util.computeStrides(filter.shape);
    const yStrides = util.computeStrides(convInfo.outShape);

    const xVals = cpuBackend.data.get(x.dataId).values as TypedArray;
    const wVals = cpuBackend.data.get(filter.dataId).values as TypedArray;

    for (let b = 0; b < convInfo.batchSize; ++b) {
      const xOffset1 = b * xStrides[0];
      const yOffset1 = b * yStrides[0];
      for (let yR = 0; yR < convInfo.outHeight; ++yR) {
        const yOffset2 = yOffset1 + yR * yStrides[1];
        const xRCorner = yR * convInfo.strideHeight - padLeft;
        for (let wR = 0; wR < filterHeight; ++wR) {
          const xR = xRCorner + wR * dilationHeight;
          if (xR < 0 || xR >= convInfo.inHeight) {
            continue;
          }
          const wOffset1 = wR * filterStrides[0];
          const xOffset2 = xOffset1 + xR * xStrides[1];
          for (let yC = 0; yC < convInfo.outWidth; ++yC) {
            const yOffset3 = yOffset2 + yC * yStrides[2];
            const xCCorner = yC * convInfo.strideWidth - padTop;
            for (let wC = 0; wC < filterWidth; ++wC) {
              const xC = xCCorner + wC * dilationWidth;
              if (xC < 0 || xC >= convInfo.inWidth) {
                continue;
              }
              const wOffset2 = wOffset1 + wC * filterStrides[1];
              const xOffset3 = xOffset2 + xC * convInfo.inChannels;
              let yOffset4 = yOffset3;
              let wOffset3 = wOffset2;
              for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                const xVal = xVals[xOffset3 + d1];
                for (let q = 0; q < chMul; ++q) {
                  yVals[yOffset4 + q] += xVal * wVals[wOffset3 + q];
                }
                yOffset4 += chMul;
                wOffset3 += chMul;
              }
            }
          }
        }
      }
    }

    const outId = cpuBackend.write(yVals, convInfo.outShape, 'float32');

    return {dataId: outId, shape: convInfo.outShape, dtype: 'float32'};
  }
};
