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

import {backend_util, Softmax, SoftmaxAttrs, SoftmaxInputs, TensorInfo, util} from '@tensorflow/tfjs-core';
import {KernelConfig} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

import {divConfig} from './Div';
import {expConfig} from './Exp';
import {maxConfig} from './Max';
import {reshapeConfig} from './Reshape';
import {subConfig} from './Sub';
import {sumConfig} from './Sum';

export const softmaxConfig: KernelConfig = {
  kernelName: Softmax,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {logits} = inputs as SoftmaxInputs;
    const {dim} = attrs as {} as SoftmaxAttrs;
    const cpuBackend = backend as MathBackendCPU;

    const axes = util.parseAxisParam([dim], logits.shape);
    const maxLogit =
        maxConfig.kernelFunc(
            {inputs: {x: logits}, backend, attrs: {reductionIndices: axes}}) as
        TensorInfo;

    const expandedShape =
        backend_util.expandShapeToKeepDim(maxLogit.shape, axes);

    const maxLogitReshaped =
        reshapeConfig.kernelFunc(
            {inputs: {x: maxLogit}, backend, attrs: {shape: expandedShape}}) as
        TensorInfo;

    const a =
        subConfig.kernelFunc(
            {inputs: {a: logits, b: maxLogitReshaped}, backend}) as TensorInfo;

    const b = expConfig.kernelFunc({inputs: {x: a}, backend}) as TensorInfo;

    const sumExp =
        sumConfig.kernelFunc({inputs: {x: b}, backend, attrs: {axis: axes}}) as
        TensorInfo;

    const sumExpReshaped =
        reshapeConfig.kernelFunc(
            {inputs: {x: sumExp}, backend, attrs: {shape: expandedShape}}) as
        TensorInfo;

    const result =
        divConfig.kernelFunc({inputs: {a: b, b: sumExpReshaped}, backend}) as
        TensorInfo;

    cpuBackend.disposeData(maxLogit.dataId);
    cpuBackend.disposeData(maxLogitReshaped.dataId);
    cpuBackend.disposeData(a.dataId);
    cpuBackend.disposeData(b.dataId);
    cpuBackend.disposeData(sumExp.dataId);
    cpuBackend.disposeData(sumExpReshaped.dataId);

    return result;
  }
};
