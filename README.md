# API_tensorflow_detection

API Object detection using SSDLite 

# Installation 
Put in a folder the API object Detection and this repository

install API Object detection
```
git clone https://github.com/tensorflow/models.git
```

don't forget to export the API in your pythonpath inside the folder models/research/

```
export PYTHONPATH=$PYTHONPATH:pwd:pwd/slim
```

# Train 
```
  python3 models/research/object_detection/train.py --logtostderr --train_dir=path/where/checkpoint/will/be/saved --pipeline_config_path=/path/to/the/config/file
```

# Evaluation

```
  python3 models/research/object_detection/eval.py --logtostderr --checkpoint_dir=/path/checkpoint/--eval_dir=path/where/evaluation/will/be/saved  --pipeline_config_path=/path/to/the/config/file
```  
