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

import {BatchMatMul, BatchMatMulAttrs, BatchMatMulInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';
import {KernelConfig} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

import {reshapeConfig} from './Reshape';

export const batchMatMulConfig: KernelConfig = {
  kernelName: BatchMatMul,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {a, b} = inputs as BatchMatMulInputs;
    const {transposeA, transposeB} = attrs as {} as BatchMatMulAttrs;
    const cpuBackend = backend as MathBackendCPU;

    assertNotComplex([a, b], 'matMul');

    const innerShapeA =
        transposeA ? a.shape[a.shape.length - 2] : a.shape[a.shape.length - 1];
    const innerShapeB =
        transposeB ? b.shape[b.shape.length - 1] : b.shape[b.shape.length - 2];

    const outerShapeA =
        transposeA ? a.shape[a.shape.length - 1] : a.shape[a.shape.length - 2];
    const outerShapeB =
        transposeB ? b.shape[b.shape.length - 2] : b.shape[b.shape.length - 1];

    const outerDimsA = a.shape.slice(0, -2);
    const outerDimsB = b.shape.slice(0, -2);
    const batchDimA = util.sizeFromShape(outerDimsA);
    const batchDimB = util.sizeFromShape(outerDimsB);

    util.assert(
        util.arraysEqual(outerDimsA, outerDimsB),
        () => `Error in matMul: outer dimensions (${outerDimsA}) and (` +
            `${outerDimsB}) of Tensors with shapes ${a.shape} and ` +
            `${b.shape} must match.`);

    util.assert(
        innerShapeA === innerShapeB,
        () => `Error in matMul: inner shapes (${innerShapeA}) and (` +
            `${innerShapeB}) of Tensors with shapes ${a.shape} and ` +
            `${b.shape} and transposeA=${transposeA}` +
            ` and transposeB=${transposeB} must match.`);

    const outShape = a.shape.slice(0, -2).concat([outerShapeA, outerShapeB]);

    const a3D = transposeA ?
        reshapeConfig.kernelFunc({
          inputs: {x: a},
          backend,
          attrs: {shape: [batchDimA, innerShapeA, outerShapeA]}
        }) as TensorInfo :
        reshapeConfig.kernelFunc({
          inputs: {x: a},
          backend,
          attrs: {shape: [batchDimA, outerShapeA, innerShapeA]}
        }) as TensorInfo;

    const b3D = transposeB ?
        reshapeConfig.kernelFunc({
          inputs: {x: b},
          backend,
          attrs: {shape: [batchDimB, outerShapeB, innerShapeB]}
        }) as TensorInfo :
        reshapeConfig.kernelFunc({
          inputs: {x: b},
          backend,
          attrs: {shape: [batchDimB, innerShapeB, outerShapeB]}
        }) as TensorInfo;

    const sharedDim = transposeA ? a3D.shape[1] : a3D.shape[2];
    const leftDim = transposeA ? a3D.shape[2] : a3D.shape[1];
    const rightDim = transposeB ? b3D.shape[1] : b3D.shape[2];
    const batchDim = a3D.shape[0];

    const a3DValues = cpuBackend.data.get(a3D.dataId).values as TypedArray;
    const b3DValues = cpuBackend.data.get(b3D.dataId).values as TypedArray;

    const a3DStrides = util.computeStrides(a3D.shape);
    const b3DStrides = util.computeStrides(b3D.shape);

    const [aBatch, aOuterStep, aInnerStep] = transposeA ?
        [a3DStrides[0], 1, a3DStrides[1]] :
        [a3DStrides[0], b3DStrides[1], 1];
    const [bInnerStep, bOuterStep, bBatch] = transposeB ?
        [1, b3DStrides[1], b3DStrides[0]] :
        [b3DStrides[1], 1, b3DStrides[0]];

    const size = leftDim * rightDim;
    const shape = [batchDim, leftDim, rightDim];

    const resVals = util.getArrayFromDType(
                        a3D.dtype, util.sizeFromShape(shape)) as TypedArray;

    const blockSize = cpuBackend.blockSize;

    for (let b = 0; b < batchDim; b++) {
      for (let i0 = 0; i0 < leftDim; i0 += blockSize) {
        for (let j0 = 0; j0 < rightDim; j0 += blockSize) {
          for (let k0 = 0; k0 < sharedDim; k0 += blockSize) {
            // for when blockSize doesn't evenly divide the input
            const iBlock = Math.min(i0 + blockSize, leftDim);
            const jBlock = Math.min(j0 + blockSize, rightDim);
            const kBlock = Math.min(k0 + blockSize, sharedDim);

            for (let i = i0; i < iBlock; i++) {
              for (let j = j0; j < jBlock; j++) {
                let sum = 0.0;

                for (let k = k0; k < kBlock; k++) {
                  sum +=
                      a3DValues[b * aBatch + i * aOuterStep + k * aInnerStep] *
                      b3DValues[k * bInnerStep + j * bOuterStep + b * bBatch];
                }
                resVals[b * size + (i * rightDim + j)] += sum;
              }
            }
          }
        }
      }
    }

    const dataId = cpuBackend.write(resVals, shape, a3D.dtype);

    const result = reshapeConfig.kernelFunc({
      inputs: {x: {dataId, shape, dtype: a3D.dtype}},
      backend,
      attrs: {shape: outShape}
    });

    cpuBackend.disposeData(a3D.dataId);
    cpuBackend.disposeData(b3D.dataId);
    cpuBackend.disposeData(dataId);

    return result;
  }
};
