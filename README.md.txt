python3 models/research/object_detection/train.py --logtostderr --train_dir=API_tensorflow_detection/training/ --pipeline_config_path=API_tensorflow_detection/model/pipeline_ssdlite.config

python3 models/research/object_detection/eval.py --logtostderr --checkpoint_dir=API_tensorflow_detection/training/model.ckpt --eval_dir=eval_model/ --pipeline_config_path=API_tensorflow_detection/model/pipeline_ssdlite.config 
