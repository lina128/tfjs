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

import {backend_util, GatherNd, GatherNdInputs, NumericDataType, TypedArray, util} from '@tensorflow/tfjs-core';
import {KernelConfig} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {reshapeConfig} from './Reshape';

export const gatherNdConfig: KernelConfig = {
  kernelName: GatherNd,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend}) => {
    const {params, indices} = inputs as GatherNdInputs;
    const cpuBackend = backend as MathBackendCPU;

    const indicesShape = indices.shape;
    const sliceRank = indicesShape[indicesShape.length - 1];

    const [resultShape, numSlices, sliceSize, strides] =
        backend_util.prepareAndValidate(params, indices);

    if (numSlices === 0) {
      const outId = cpuBackend.write([], resultShape, params.dtype);
      return {dataId: outId, resultShape, dtype: params.dtype};
    }

    const sliceShape = [numSlices, sliceSize];
    const paramsSize = util.sizeFromShape(params.shape);
    const resultSize = util.sizeFromShape(sliceShape);
    const resVals = util.getTypedArrayFromDType(
        params.dtype as NumericDataType, resultSize);

    const indicesData =
        cpuBackend.data.get(indices.dataId).values as TypedArray;
    const xData = cpuBackend.data.get(params.dataId).values as TypedArray;

    for (let i = 0; i < numSlices; i++) {
      const index = [];
      let flattenIndex = 0;
      for (let j = 0; j < sliceRank; j++) {
        const dim = indicesData[i * sliceRank + j];
        flattenIndex += dim * strides[j];
        index.push(dim);
      }
      if (flattenIndex < 0 || flattenIndex >= paramsSize / sliceSize) {
        throw new Error(
            `Invalid indices: ${index} does not index into ${params.shape}`);
      }

      for (let k = 0; k < sliceSize; k++) {
        resVals[i * sliceSize + k] = xData[flattenIndex * sliceSize + k];
      }
    }

    const outId = cpuBackend.write(resVals, resultShape, params.dtype);

    return {dataId: outId, shape: resultShape, dtype: params.dtype};
  }
};
