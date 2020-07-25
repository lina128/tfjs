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

import {backend_util, KernelConfig, KernelFunc, Sum, SumAttrs, SumInputs, TensorInfo, TypedArray, upcastType, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

import {castConfig} from './Cast';
import {reshapeConfig} from './Reshape';
import {transposeConfig} from './Transpose';

const sum: KernelFunc = ({inputs, backend, attrs}) => {
  const {x} = inputs as SumInputs;
  const {axis, keepDims} = attrs as {} as SumAttrs;
  const cpuBackend = backend as MathBackendCPU;

  assertNotComplex(x, 'sum');

  let $x: TensorInfo;
  if (x.dtype === 'bool') {
    $x = castConfig.kernelFunc(
             {inputs: {x}, backend, attrs: {dtype: 'int32'}}) as TensorInfo;
  } else {
    $x = castConfig.kernelFunc(
             {inputs: {x}, backend, attrs: {dtype: x.dtype}}) as TensorInfo;
  }

  const axes = util.parseAxisParam(axis, $x.shape);

  const permutation = backend_util.getAxesPermutation(axes, $x.shape.length);

  let reductionAxes = axes;
  let permutedX = $x;

  if (permutation != null) {
    permutedX = transposeConfig.kernelFunc(
                    {inputs: {x: $x}, backend, attrs: {perm: permutation}}) as
        TensorInfo;

    reductionAxes =
        backend_util.getInnerMostAxes(reductionAxes.length, $x.shape.length);
  }

  backend_util.assertAxesAreInnerMostDims(
      'sum', reductionAxes, permutedX.shape.length);

  const [outShape, reduceShape] =
      backend_util.computeOutAndReduceShapes(permutedX.shape, reductionAxes);

  const resultDtype = upcastType(permutedX.dtype, 'int32');

  const vals = util.makeZerosTypedArray(
                   util.sizeFromShape(outShape), resultDtype) as TypedArray;

  const reduceSize = util.sizeFromShape(reduceShape);

  const aVals = cpuBackend.data.get(permutedX.dataId).values as TypedArray;

  for (let i = 0; i < vals.length; ++i) {
    const offset = i * reduceSize;
    let sum = 0;
    for (let j = 0; j < reduceSize; ++j) {
      sum += aVals[offset + j];
    }
    vals[i] = sum;
  }

  const outId = cpuBackend.write(vals, outShape, resultDtype);
  const out = {dataId: outId, shape: outShape, dtype: resultDtype};

  cpuBackend.disposeData($x.dataId);
  cpuBackend.disposeData(permutedX.dataId);

  if (keepDims) {
    const newShape = backend_util.expandShapeToKeepDim(outShape, axes);
    const outReshaped = reshapeConfig.kernelFunc(
        {inputs: {x: out}, backend, attrs: {shape: newShape}});

    cpuBackend.disposeData(outId);

    return outReshaped;
  } else {
    return out;
  }
};

export const sumConfig: KernelConfig = {
  kernelName: Sum,
  backendName: 'cpu',
  kernelFunc: sum
};
