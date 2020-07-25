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

import {BinaryInputs, KernelConfig, KernelFunc, Sub, TensorInfo, TypedArray, upcastType} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {broadcastedBinaryComplexOp, broadcastedBinaryOp} from '../utils/binary_utils';

import {castConfig} from './Cast';
import {complexConfig} from './Complex';

const sub_: KernelFunc = ({inputs, backend}) => {
  const {a, b} = inputs as BinaryInputs;
  const cpuBackend = backend as MathBackendCPU;

  if (a.dtype === 'complex64' || b.dtype === 'complex64') {
    const $a = castConfig.kernelFunc(
                   {inputs: {x: a}, backend, attrs: {dtype: 'complex64'}}) as
        TensorInfo;

    const $b = castConfig.kernelFunc(
                   {inputs: {x: b}, backend, attrs: {dtype: 'complex64'}}) as
        TensorInfo;

    const aRealId = cpuBackend.data.get($a.dataId).complexInfo.real.dataId;
    const aImagId = cpuBackend.data.get($a.dataId).complexInfo.imag.dataId;
    const bRealId = cpuBackend.data.get($b.dataId).complexInfo.real.dataId;
    const bImagId = cpuBackend.data.get($b.dataId).complexInfo.imag.dataId;

    const aRealVals = cpuBackend.data.get(aRealId).values as TypedArray;
    const aImagVals = cpuBackend.data.get(aImagId).values as TypedArray;
    const bRealVals = cpuBackend.data.get(bRealId).values as TypedArray;
    const bImagVals = cpuBackend.data.get(bImagId).values as TypedArray;

    const {realVals, imagVals, shape} = broadcastedBinaryComplexOp(
        aRealVals, aImagVals, bRealVals, bImagVals, a.shape, b.shape,
        (aReal: number, aImag: number, bReal: number, bImag: number) => {
          return {real: aReal - bReal, imag: aImag - bImag};
        });

    const realId = cpuBackend.write(realVals, shape, 'float32');
    const imagId = cpuBackend.write(imagVals, shape, 'float32');

    const dataId = complexConfig.kernelFunc({
      inputs: {
        real: {dataId: realId, shape, dtype: 'float32'},
        imag: {dataId: imagId, shape, dtype: 'float32'}
      },
      backend
    });

    cpuBackend.disposeData($a.dataId);
    cpuBackend.disposeData($b.dataId);
    cpuBackend.disposeData(realId);
    cpuBackend.disposeData(imagId);

    return {dataId, shape, dtype: 'complex64'};
  }

  const aVals = cpuBackend.data.get(a.dataId).values as TypedArray;
  const bVals = cpuBackend.data.get(b.dataId).values as TypedArray;
  const dtype = upcastType(a.dtype, b.dtype);

  const {values, shape} = broadcastedBinaryOp(
      aVals, bVals, a.shape, b.shape, dtype,
      (aValue, bValue) => aValue - bValue);

  const dataId = cpuBackend.write(values, shape, dtype);

  return {dataId, shape, dtype};
};

export const subConfig: KernelConfig = {
  kernelName: Sub,
  backendName: 'cpu',
  kernelFunc: sub_
};
