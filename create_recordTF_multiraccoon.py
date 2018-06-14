import os
import numpy as np
import tensorflow as tf
import cv2
from object_detection.utils import dataset_util
from optparse import OptionParser

#flags = tf.app.flags
#flags.DEFINE_string('output_path', 'aaa', 'Path to output TFRecord')
#FLAGS = flags.FLAGS
def get_data_raccoon(input_path, path='', channels=4, formats='png'):
    found_bg = False
    all_imgs = {}

    classes_count = {}
    classes_count_train = {}
    classes_count_test = {}

    class_mapping = {}

    visualise = True

    with open(input_path,'r') as f:

        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            if channels==1:
                (filename1,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
                filename = [filename1]
            elif channels==2:
                (filename1,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
                filename = [filename1, filename1]
            elif channels==3:
                (filename1,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
                filename = [filename1, filename1, filename1]
            elif channels==4:
                (filename1,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
                filename = [filename1, filename1, filename1, filename1]
            elif channels==5:
                (filename1,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
                filename = [filename1, filename1, filename1, filename1, filename1]
            elif channels==6:
                (filename1,width, height, x1,y1,x2,y2,class_name, imageset) = line_split
                filename = [filename1, filename1, filename1, filename1, filename1, filename1]
            else:
                print('channels > 6 ; not Implemmented yet')
                exit()
            
            basename = filename[0]
            
            if imageset not in ('training', 'testing'):
                print('Imageset of ' + filename + 'is neither training nor validation, skipping image')
                continue

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if imageset == 'training':
                if class_name not in classes_count_train:
                    classes_count_train[class_name] = 1
                else:
                    classes_count_train[class_name] += 1
            elif imageset == 'testing':
                if class_name not in classes_count_test:
                    classes_count_test[class_name] = 1
                else:
                    classes_count_test[class_name] += 1


            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if basename not in all_imgs:
                all_imgs[basename] = {}
                #print(filename)
                #img = cv2.imread(filename)
                #(rows,cols) = img.shape[:2]
                all_imgs[basename]['filepath'] = [path+name for name in filename]
                all_imgs[basename]['width'] = int(width)
                all_imgs[basename]['height'] = int(height)
                all_imgs[basename]['bboxes'] = []
                all_imgs[basename]['imageset'] = imageset
                all_imgs[basename]['format'] = formats
            
            all_imgs[basename]['bboxes'].append({'class': class_name, 'xmin': int(x1), 'xmax': int(x2), 'ymin': int(y1), 'ymax': int(y2)})


        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        for key in class_mapping.keys():
            if key == 'bg':
                class_mapping[key] = 0
            else:
                class_mapping[key] += 1

        if len(classes_count_train.keys()) == len(classes_count_test.keys()):
            print('Number of classes : ', len(classes_count_train.keys()))
            return all_data, classes_count, class_mapping, classes_count_train, classes_count_test
        else:
            return None, None, None, None, None


def create_one_tf_example(data, class_map, path='', channels=4):
    '''
      Creates a tf.Example proto from sample cat image.

      Args:
        data with a dictionnary containing path, width etc

      Returns:
        example: The created tf.Example.
    '''

    height = data['height']
    width = data['width']
    filename = data['filepath']
    format_img = data['format']
    #with tf.gfile.GFile(path+filename[0], 'r') as fid:
        #print(fid.shape)
        #encoded_image_data = fid.read()
        
    # we consider that each image contain one channel
    for i in range(len(filename)):
        temp = np.mean(cv2.imread(filename[i]), axis=-1)
        if temp is None:
            print("can't open image, check path parameters ! ")
            return
        temp = np.expand_dims(temp, axis=-1)
        if i == 0:
            img = temp
        else:
            img = np.concatenate([img, temp], axis=-1)
    #if img.shape != (1920, 1920, 4):
     #   print(img.shape)
    img = img.astype(np.uint8)
    img_encoded = img.tostring()
    #filename = filename.encode('utf8')
    if format_img == 'png':
        image_format = b'png'
    elif format_img == 'jpg':
        image_format = b'jpg'
    else:
        print('Error format retry')
        
        
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    
    for box in data['bboxes']:
        xmins.append(box['xmin']/float(width))
        xmaxs.append(box['xmax']/float(width))
        ymins.append(box['ymin']/float(height))
        ymaxs.append(box['ymax']/float(height))
        
        classes_text.append(box['class'].encode('utf8'))
        classes.append(class_map[box['class']])
    
    # with tf.gfile.GFile(os.path.join(path, '{}'.format(filename)), 'rb') as fid:
    #print(filename)
    #print(path+filename)
    #print(cv2.imread((path+filename).replace('\\','/')))
    
      
    feat = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      #'image/filename': dataset_util.bytes_feature(filename),
      #'image/source_id': dataset_util.bytes_feature(filename),
      'image/channels': dataset_util.int64_feature(channels),
      'image/encoded': dataset_util.bytes_feature(img_encoded),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes)
      }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feat))
    return tf_example

def write_map(class_map, filename_output='map_label.pbtxt'):
    '''
        write map label for tensorflow API detection
        input : dictionnary, key:class, value:number
        
    '''
    dicbykey = {value:key for (key, value) in class_map.items()}
    print(dicbykey)
    sorted_number = sorted(dicbykey.keys())
    with open(filename_output, 'w') as f:
        size = len(dicbykey)
        for i, num in enumerate(sorted_number):
            if i == size:
                f.write('item {\n name: "'+dicbykey[num]+'"\n  id: '+str(num)+'\n}')
            else:
                f.write('item {\n name: "'+dicbykey[num]+'"\n  id: '+str(num)+'\n}\n')
        
def __main__(filename = 'raccoon_dataset.txt', output_label = 'raccoon_label.pbtxt', output_train = 'train.record', output_test = 'test.record', num_channels=4, path='', formats='png'):
    '''
        convert from filename image to record
    '''

    
    
    #path = os.getcwd()+'\\'+'raccoon\\'
    all_data, classes_count, class_mapping, classes_count_train, classes_count_test = get_data_raccoon(filename, path=path, channels=num_channels, formats=formats)
    print(class_mapping)
    print(len(all_data))
    writer_train = tf.python_io.TFRecordWriter(output_train)
    writer_test = tf.python_io.TFRecordWriter(output_test)
    cpt_train = 0
    cpt_test = 0
    for data in all_data:
        if data['imageset'] == 'training':
            cpt_train = cpt_train + 1
            tf_example = create_one_tf_example(data, class_mapping, path=path, channels=num_channels)
            writer_train.write(tf_example.SerializeToString())
        elif data['imageset'] == 'testing':
            cpt_test = cpt_test + 1
            tf_example = create_one_tf_example(data, class_mapping, path=path, channels=num_channels)
            writer_test.write(tf_example.SerializeToString())
#
    writer_train.close()
    writer_test.close()
    print('Successfully created the TFRecords: {}'.format(os.getcwd() + output_train))
    print('Successfully created the TFRecords: {}'.format(os.getcwd() + output_test))
    write_map(class_mapping, output_label)
    print('Successfully created the label file: {}'.format(os.getcwd() + output_label))
    print('number train/test ', cpt_train, cpt_test)

##### MAIN ######
parser = OptionParser()
parser.add_option('-f', "--filename", dest="filename", help='dataset filename', default='raccoon_dataset.txt')
parser.add_option("--train_output", dest="train_output", help="name of the train record file", default='trainraccoon4.record')
parser.add_option("--test_output", dest="test_output", help="name of the test record file", default='testraccoon4.record')
parser.add_option("--label_output", dest="label_output", help="name of the label pbtxt file", default='labelraccoon4.pbtxt')
parser.add_option("-c", "--channels", dest="channels", help="name of the train record file", default=4)
parser.add_option("-p", "--path", dest="path", help="path where images are", default='../dataset/raccoon/')
parser.add_option("--format", dest="format", help="path where images are", default='png')
(options, args) = parser.parse_args()
filename = options.filename
output_train = options.train_output
output_test = options.test_output
output_label = options.label_output
num_channels = int(options.channels)
path = options.path
__main__(filename=filename, output_train = output_train, output_test=output_test, output_label=output_label, num_channels=num_channels, path=path, formats=options.format)
