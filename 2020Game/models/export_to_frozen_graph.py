# Takes a directory saved by the best_exporter code and converts it to a frozen graph
import tensorflow as tf
from object_detection.protos import pipeline_pb2
import os
from tensorflow.python.tools import freeze_graph  # pylint: disable=g-direct-tensorflow-import
from google.protobuf import text_format
from object_detection import exporter
from object_detection.builders import model_builder
from object_detection.utils import config_util

#pb_saved_model = "/home/ubuntu/tensorflow_workspace/2020Game/models/trained_retinanet/export/best_exporter/1591324092"
pb_saved_model_path = "/home/ubuntu/tensorflow_workspace/2020Game/models/trained_ssd_mobilenet_v2_coco_focal_loss/export/best_exporter/1590476859"
model_config_file = "/home/ubuntu/tensorflow_workspace/2020Game/models/model/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config"
BOXES_NAME='detection_boxes'
CLASSES_NAME='detection_classes'
SCORES_NAME='detection_scores'
NUM_DETECTIONS_NAME='num_detections'
OUTPUT_NAMES = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]


def export_inf_graph(input_type,
                    detection_model,
                    input_saved_model_path,
                    additional_output_tensor_names=None,
                    input_shape=None,
                    output_collection_name='inference_op',
                    graph_hook_fn=None,
                    write_inference_graph=False,
                    temp_checkpoint_prefix=''):

    """Export helper."""
    output_directory = os.path.join(input_saved_model_path, 'output')
    tf.gfile.MakeDirs(output_directory)
    frozen_graph_path = os.path.join(output_directory, 'frozen_inference_graph.pb')
    saved_model_path = os.path.join(output_directory, 'saved_model')
    model_path = os.path.join(output_directory, 'model.ckpt')

    outputs, placeholder_tensor = exporter.build_detection_graph(
      input_type=input_type,
      detection_model=detection_model,
      input_shape=input_shape,
      output_collection_name=output_collection_name,
      graph_hook_fn=graph_hook_fn)

    exporter.profile_inference_graph(tf.get_default_graph())
    saver_kwargs = {}

    if additional_output_tensor_names is not None:
        output_node_names = ','.join(outputs.keys()+additional_output_tensor_names)
    else:
        output_node_names = ','.join(outputs.keys())

    print input_saved_model_path
    frozen_graph_def = freeze_graph.freeze_graph_with_def_protos(
            input_graph_def=tf.get_default_graph().as_graph_def(),
            input_saver_def=None,
            input_checkpoint=None,
            output_node_names=output_node_names,
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=frozen_graph_path,
            clear_devices=True,
            initializer_nodes='',
            input_saved_model_dir=input_saved_model_path,
            saved_model_tags=["serve"])

    write_saved_model(saved_model_path, frozen_graph_def,
                                        placeholder_tensor, outputs)


input_shape = None
score_threshold = None
batch_size = 1
output_collection_name='inference_op'
additional_output_tensor_names = None
write_inference_graph = False

# parse pipeline config from file
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with open(model_config_file, 'r') as f:
    text_format.Merge(f.read(), pipeline_config, allow_unknown_extension=True)

# override some pipeline_config parameters
if pipeline_config.model.HasField('ssd'):
    pipeline_config.model.ssd.feature_extractor.override_base_feature_extractor_hyperparams = True
    if score_threshold is not None:
        pipeline_config.model.ssd.post_processing.batch_non_max_suppression.score_threshold = score_threshold    
    if input_shape is not None:
        pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height = input_shape[0]
        pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width = input_shape[1]
elif pipeline_config.model.HasField('faster_rcnn'):
    if score_threshold is not None:
        pipeline_config.model.faster_rcnn.second_stage_post_processing.score_threshold = score_threshold
    if input_shape is not None:
        pipeline_config.model.faster_rcnn.image_resizer.fixed_shape_resizer.height = input_shape[0]
        pipeline_config.model.faster_rcnn.image_resizer.fixed_shape_resizer.width = input_shape[1]

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

input_shape=[batch_size, None, None, 3]
detection_model = model_builder.build(pipeline_config.model, is_training=False)

graph_rewriter_fn = None
if pipeline_config.HasField('graph_rewriter'):
    graph_rewriter_config = pipeline_config.graph_rewriter
    graph_rewriter_fn = graph_rewriter_builder.build(graph_rewriter_config, is_training=False)

export_inf_graph('image_tensor',
        detection_model,
        pb_saved_model_path,
        additional_output_tensor_names,
        input_shape,
        output_collection_name,
        graph_hook_fn=graph_rewriter_fn,
        write_inference_graph=write_inference_graph)
pipeline_config.eval_config.use_moving_averages = False
config_util.save_pipeline_config(pipeline_config, pb_saved_model_path)

"""
_graph = tf.Graph()
with _graph.as_default():
    _sess = tf.Session(graph=_graph)
    model = tf.saved_model.loader.load(_sess, ["serve"], pb_saved_model_path)
    #print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))
    graphdef = tf.get_default_graph().as_graph_def()
    frozen_graph = tf.graph_util.convert_variables_to_constants(_sess,graphdef, OUTPUT_NAMES)
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

#with tf.gfile.GFile(os.path.join(pb_saved_model_path, "frozen_inference_graph.pb"), "wb") as f:
    #f.write(frozen_graph.SerializeToString())

input_type = 'image_tensor'
output_collection_name='inference_op'
graph_hook_fn=None
outputs, placeholder_tensor = exporter.build_detection_graph(
      input_type,
      detection_model,
      input_shape,
      output_collection_name,
      graph_hook_fn)

exporter.write_saved_model(os.path.join(pb_saved_model_path, 'output'),
        frozen_graph,
        placeholder_tensor,
        outputs)
config_util.save_pipeline_config(pipeline_config, os.path.join('output', pb_saved_model_path))

print(placeholder_tensor)
print(outputs)
"""
