import tensorflow as tf
import cv2
import json
import os
import numpy as np
import albumentations as alb
from matplotlib import pyplot as plt

class NeuralNetworkInterface:
    augmentor = None
    def __init__(self):
        tf.get_logger().setLevel('ERROR')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.images = None
        self.image_generator = None
        self.batch_size = 1

        self.augmentor = alb.Compose([alb.RandomCrop(width=1024, height=1024), 
                         alb.HorizontalFlip(p=0.5), 
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2), 
                         alb.RGBShift(p=0.2), 
                         alb.VerticalFlip(p=0.5)], 
                       bbox_params=alb.BboxParams(format='albumentations', 
                                                  label_fields=['class_labels']))
        # Initialize any necessary variables or objects here
        pass

    def loadTrainingImagesIntoPipeline(self, image_path):
        # Load images into the pipeline here
        self.images = tf.data.Dataset.list_files(image_path + '/*.jpg', shuffle=False)
        pass

    def loadImagesInDirectory(self, image_path):
        # Load images into the pipeline here
        return tf.data.Dataset.list_files(image_path + '/*.jpg', shuffle=False)
        

    def loadImage(self, x):
        byte_img = tf.io.read_file(x)
        img = tf.io.decode_jpeg(byte_img)
        #call with images.map()
        return img
    
    def batchImages(self, batch_size):
        # Batch images here
        self.batch_size = batch_size
        self.images = self.images.map(self.loadImage)
        self.image_generator = self.images.batch(batch_size).as_numpy_iterator()
        pass

    def showNextBatch(self):
        # Show the next batch of images here
        batch = self.image_generator.next()
        
        fig, ax = plt.subplots(ncols=self.batch_size, figsize=(60,60))
        for idx, image in enumerate(batch):
            ax[idx].imshow(image)
        plt.show()

    def moveMatchingLabels(self, data_path):
        for folder in ['train', 'val', 'test']:
            for file in os.listdir(os.path.join(data_path, folder, 'images')):
                filename = file.split('.')[0]+'.json'
                existing_filepath = os.path.join(data_path, 'labels', filename)
                if os.path.exists(existing_filepath):
                    new_filepath = os.path.join(data_path, folder, 'labels', filename)
                    os.replace(existing_filepath, new_filepath)

    def pullJsonData (self, path):
        with open(path, 'r') as f:
            return json.load(f)
        
    def loadLabels(self, label_path):
        label = self.pullJsonData(label_path)
        return [label['class'], label['bbox']]

    def extractCoordinates(self, retJson):
        coords = [0,0,0,0]
        coords[0] = retJson['shapes'][0]['points'][0][0]
        coords[1] = retJson['shapes'][0]['points'][0][1]
        coords[2] = retJson['shapes'][0]['points'][1][0]
        coords[3] = retJson['shapes'][0]['points'][1][1]
        print("Coords: " + str(coords))
        return coords
    
    def changeCoordinateFormatToAlbumentations(self, coords, img_sizex, img_sizey):
        #change from pascal to albumentation

        cordlist = list(np.divide(coords, [img_sizex, img_sizey, img_sizex, img_sizey]))
        print("Cordlist: " + str(cordlist))
        return cordlist
    
    def augmentImage(self, img, coords, label):
        augmented = self.augmentor(image=img, bboxes=[coords], class_labels=[label])
        return augmented
    
    def buildAugmentations(self, data_path):
        for partition in ['train','test','val']: 
            for image in os.listdir(os.path.join(data_path, partition, 'images')):
                img = cv2.imread(os.path.join(data_path, partition, 'images', image))

                coords = [0,0,0.00001,0.00001]
                label_path = os.path.join(data_path, partition, 'labels', f'{image.split(".")[0]}.json')
                if os.path.exists(label_path):
                    label = self.pullJsonData(label_path)
                    coords = self.extractCoordinates(label)
                    coords = self.changeCoordinateFormatToAlbumentations(coords, 1920, 1080)

                for x in range(60):
                    augmented = self.augmentImage(img, coords, 'face')
                    cv2.imwrite(os.path.join(data_path, 'aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                    annotation = {}
                    annotation['image'] = image

                    if os.path.exists(label_path):
                        if len(augmented['bboxes']) == 0: 
                            annotation['bbox'] = [0,0,0,0]
                            annotation['class'] = 0 
                        else: 
                            annotation['bbox'] = augmented['bboxes'][0]
                            annotation['class'] = 1
                    else: 
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0 


                    with open(os.path.join(data_path, 'aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                        json.dump(annotation, f)

    def template(self, data_path):
        train_images = self.loadImagesInDirectory(os.path.join(data_path, 'aug_data', 'train', 'images'))
        train_images = train_images.map(self.loadImage)
        train_images = train_images.map(lambda x: tf.image.resize(x, (240, 240)))
        train_images = train_images.map(lambda x: x / 255)

        test_images = self.loadImagesInDirectory(os.path.join(data_path, 'aug_data', 'test', 'images'))
        test_images = test_images.map(self.loadImage)
        test_images = test_images.map(lambda x: tf.image.resize(x, (240, 240)))
        test_images = test_images.map(lambda x: x / 255)

        val_images = self.loadImagesInDirectory(os.path.join(data_path, 'aug_data', 'val', 'images'))
        val_images = val_images.map(self.loadImage)
        val_images = val_images.map(lambda x: tf.image.resize(x, (240, 240)))
        val_images = val_images.map(lambda x: x / 255)

        train_labels = tf.data.Dataset.list_files(data_path + '\\aug_data\\train\\labels\\*.json', shuffle=False)
        train_labels = train_labels.map(lambda x: tf.py_function(self.loadLabels, [x], [tf.uint8, tf.float16]))
        test_labels = tf.data.Dataset.list_files(data_path + '\\aug_data\\test\\labels\\*.json', shuffle=False)
        test_labels = test_labels.map(lambda x: tf.py_function(self.loadLabels, [x], [tf.uint8, tf.float16]))
        val_labels = tf.data.Dataset.list_files(data_path + '\\aug_data\\val\\labels\\*.json', shuffle=False)
        val_labels = val_labels.map(lambda x: tf.py_function(self.loadLabels, [x], [tf.uint8, tf.float16]))

        print (len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels))

        train = tf.data.Dataset.zip((train_images, train_labels))
        train = train.shuffle(5000)
        train = train.batch(8)
        train = train.prefetch(4)
        test = tf.data.Dataset.zip((test_images, test_labels))
        test = test.shuffle(1300)
        test = test.batch(8)
        test = test.prefetch(4)
        val = tf.data.Dataset.zip((val_images, val_labels))
        val = val.shuffle(1000)
        val = val.batch(8)
        val = val.prefetch(4)

        print(train.as_numpy_iterator().next()[0].shape)

