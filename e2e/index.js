import 'regenerator-runtime/runtime'
import '@tensorflow/tfjs-backend-cpu';
import {loadGraphModel} from '@tensorflow/tfjs-converter';
import {tensor1d, tensor2d} from '@tensorflow/tfjs-core';

async function main() {
  const model = await loadGraphModel('http://localhost:8080/model.json');
  // global map destination places
  const tensor1 = tensor1d(['a'], 'string');        // values
  const tensor13 = tensor2d([0], [1, 1], 'int32');  // indices
  const tensor3 = tensor1d([1], 'int32');           // shape

  // probability visit
  const tensor11 = tensor1d([0.5]);                      // values
  const tensor19 = tensor2d([[0, 0]], [1, 2], 'int32');  // indices
  const tensor4 = tensor1d([1, 1], 'int32');             // shape

  // destination places
  const tensor7 = tensor1d(['a'], 'string');        // values
  const tensor12 = tensor2d([0], [1, 1], 'int32');  // indices
  const tensor5 = tensor1d([1], 'int32');           // shape

  // global map traffic portion
  const tensor10 = tensor1d([0.5]);                 // values
  const tensor15 = tensor2d([0], [1, 1], 'int32');  // indices
  const tensor6 = tensor1d([1], 'int32');           // shape

  // curr session visits count
  const tensor8 = tensor1d([0], 'int32');           // values;
  const tensor16 = tensor2d([0], [1, 1], 'int32');  // indices
  const tensor9 = tensor1d([1], 'int32');           // shape

  // avg visits per session
  const tensor2 = tensor1d([0.5]);                  // values;
  const tensor17 = tensor2d([0], [1, 1], 'int32');  // indices
  const tensor14 = tensor1d([1], 'int32');          // shape

  // cur location
  const tensor18 = tensor1d(['a'], 'string');

  result = await model.executeAsync({
    'global_map_destination_places/values': tensor1,
    'global_map_destination_places/indices': tensor13,
    'global_map_destination_places/shape': tensor3,
    'probability_visit/values': tensor11,
    'probability_visit/indices': tensor19,
    'probability_visit/shape': tensor4,
    'destination_places/values': tensor7,
    'destination_places/indices': tensor12,
    'destination_places/shape': tensor5,
    'global_map_traffic_portion/values': tensor10,
    'global_map_traffic_portion/indices': tensor15,
    'global_map_traffic_portion/shape': tensor6,
    'cur_session_visits_count/values': tensor8,
    'cur_session_visits_count/indices': tensor16,
    'cur_session_visits_count/shape': tensor9,
    'avg_visits_per_session_count/values': tensor2,
    'avg_visits_per_session_count/indices': tensor17,
    'avg_visits_per_session_count/shape': tensor14,
    'cur_location': tensor18,
  });
}

main();
