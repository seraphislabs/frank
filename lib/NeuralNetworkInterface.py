import tensorflow as tf
import cv2
import json
import os
import numpy as np
import albumentations as alb
from matplotlib import pyplot as plt
from pathlib import Path
import random

class FaceTracker(tf.keras.Model):
    def __init__(self, eyetracker,  **kwargs): 
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt
    
    def train_step(self, batch, **kwargs): 
        
        X, y = batch
        
        with tf.GradientTape() as tape: 
            classes, coords = self.model(X, training=True)
            
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            
            total_loss = batch_localizationloss+0.5*batch_classloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
    
    def test_step(self, batch, **kwargs): 
        X, y = batch
        
        classes, coords = self.model(X, training=False)
        
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss+0.5*batch_classloss
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)

class NeuralNetworkTrainingInterface:
    augmentor = None
    data_sub_folder = None
    data_sub_full_path = None
    stucture_exsists = False
    model_train_images = None

    def __init__(self, _data_sub_folder):
        self._data_sub_folder = _data_sub_folder

        # Set up augmentor for image variation duplication.
        self.augmentor = alb.Compose([alb.RandomCrop(width=1024, height=1024), 
                                      alb.HorizontalFlip(p=0.5), 
                                      alb.RandomBrightnessContrast(p=0.2),
                                      alb.RandomGamma(p=0.2), 
                                      alb.RGBShift(p=0.2), 
                                      alb.VerticalFlip(p=0.5)], 
                                     bbox_params=alb.BboxParams(format='albumentations', 
                                                               label_fields=['class_labels']))
                
        # Determine the absolute path of the data subfolder.
        current_directory = Path(__file__).resolve().parent
        parent_directory = current_directory.parent
        self.data_sub_full_path = parent_directory / "data" / self._data_sub_folder

        if not (os.path.exists(self.data_sub_full_path)):
            os.makedirs(self.data_sub_full_path)

        # Check if the required folder structure exists.
        self.structure_exists = self.checkCreateFileStructure()
        
        if (self.structure_exists):
            print("** NNI Trainer: Folder structure exists. Interface Initialized.")
        else:
            print("** NNI Trainer: Folder structure created. Interface Initialized.")
            self.structure_exists = True

    def checkCreateFileStructure(self):
        # Define the required folder structure
        required_structure = {
            'input_images': [],
            'input_labels': [],
            'model': ['images', 'labels', 'train', 'evaluate', 'test']
        }

        # Define the subfolders for 'train', 'evaluate', and 'test'
        subfolders = ['images', 'labels']

        # Variable to keep track of whether any folders were created
        folders_created = False

        # Check each folder in the required structure
        for folder, subdirs in required_structure.items():
            folder_path = os.path.join(self.data_sub_full_path, folder)

            # Check and create the main folder if necessary
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                folders_created = True

            # Check and create subdirectories
            for subdir in subdirs:
                subdir_path = os.path.join(folder_path, subdir)

                # For 'model' subdirectories, check further subfolders
                if folder == 'model' and subdir in ['train', 'evaluate', 'test']:
                    for sub_subdir in subfolders:
                        sub_subdir_path = os.path.join(subdir_path, sub_subdir)
                        if not os.path.exists(sub_subdir_path):
                            os.makedirs(sub_subdir_path)
                            folders_created = True

                elif not os.path.exists(subdir_path):
                    os.makedirs(subdir_path)
                    folders_created = True

        return not folders_created
    
    def countContents(self, folder):
        folder_path = os.path.join(self.data_sub_full_path, folder)
        contents = os.listdir(folder_path)
        return contents
    
    def getJpgsInFolder(self, folder):
        folder_path = os.path.join(self.data_sub_full_path, folder)
        contents = os.listdir(folder_path)
        jpgs = []
        for content in contents:
            if content.lower().endswith('.jpg'):
                full_path = os.path.join(folder_path, content)  # Join the folder path with the file name
                jpgs.append(full_path)  # Add the full path to the list
        return jpgs

    def getJsonsInFolder(self, folder):
        folder_path = os.path.join(self.data_sub_full_path, folder)
        contents = os.listdir(folder_path)
        jsons = []
        for content in contents:
            if content.lower().endswith('.json'):
                full_path = os.path.join(folder_path, content)  # Join the folder path with the file name
                jsons.append(full_path)  # Add the full path to the list
        return jsons
    
    def filter_matching_files(self, img_paths, json_paths):
        # Extract the base filenames without extensions
        img_basenames = {os.path.splitext(os.path.basename(img))[0] for img in img_paths}
        json_basenames = {os.path.splitext(os.path.basename(json))[0] for json in json_paths}

        # Find common basenames
        common_basenames = img_basenames.intersection(json_basenames)

        # Filter the original lists to include only files with matching basenames
        filtered_img_paths = [img for img in img_paths if os.path.splitext(os.path.basename(img))[0] in common_basenames]
        filtered_json_paths = [json for json in json_paths if os.path.splitext(os.path.basename(json))[0] in common_basenames]
        return filtered_img_paths, filtered_json_paths

    def processInputImages(self):
        imgs = self.getJpgsInFolder('input_images')
        jsons = self.getJsonsInFolder('input_labels')
        
        imgs, jsons = self.filter_matching_files(imgs, jsons)
        print("** NNI Trainer: Processing Input Images #".format(len(imgs)))

        #self.move_files(imgs, os.path.join(self.data_sub_full_path, 'model', 'img_dump'))
        #self.move_files(jsons, os.path.join(self.data_sub_full_path, 'model', 'lbl_dump'))

        for img_path in imgs:
            img = cv2.imread(img_path)
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            coords = [0,0,0.00001,0.00001]
            label_path = os.path.join(self.data_sub_full_path, 'input_labels', f'{img_name}.json')

            if not os.path.exists(label_path):
                print("=> No label found for image {}".format(img_name))
                continue

            label = json.load(open(label_path, 'r'))

            if label['shapes'] == []:
                print("=> No label found for image {}".format(img_name))
                continue

            coords = self.getLabelInformation(label, img)

            for x in range(60):
                augmented = self.augmentor(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join(self.data_sub_full_path, 'model', 'images', f'{img_name}.{x}.jpg'), augmented['image'])

                annotation = {}
                annotation['image'] = img_path

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


                with open(os.path.join(self.data_sub_full_path, 'model', 'labels', f'{img_name}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

                percent_complete = ((x + 1) / 60) * 100
                print(f"\rProgress: {percent_complete:.2f}%", end='')

        print("=> Done.")
        return True
    
    def unloadModelImages(self):
        self.empty_directory(os.path.join(self.data_sub_full_path, 'model', 'train', 'images'))
        self.empty_directory(os.path.join(self.data_sub_full_path, 'model', 'train', 'labels'))
        self.empty_directory(os.path.join(self.data_sub_full_path, 'model', 'evaluate', 'images'))
        self.empty_directory(os.path.join(self.data_sub_full_path, 'model', 'evaluate', 'labels'))
        self.empty_directory(os.path.join(self.data_sub_full_path, 'model', 'test', 'images'))
        self.empty_directory(os.path.join(self.data_sub_full_path, 'model', 'test', 'labels'))

    def unloadAugmentations(self):
        self.empty_directory(os.path.join(self.data_sub_full_path, 'model', 'images'))
        self.empty_directory(os.path.join(self.data_sub_full_path, 'model', 'labels'))

    def loadModelImages(self):
        print("** NNI Trainer: Loading Model Images")

        if os.listdir(os.path.join(self.data_sub_full_path, 'model', 'train', 'images')):
            print("=> Model images already loaded.")
            return True

        jpgs = self.getJpgsInFolder(os.path.join('model', 'images'))

        trainJpgs, evaluteJpgs, testJpgs = self.splitListForLoading(jpgs, 70, 20, 10)
        totalJpgs = len(trainJpgs) + len(evaluteJpgs) + len(testJpgs)
        jpgCount = 0

        for jpg in trainJpgs:
            jpgCount += 1
            img = cv2.imread(jpg)
            img_name = os.path.splitext(os.path.basename(jpg))[0]
            label_path = os.path.join(self.data_sub_full_path, 'model', 'labels', f'{img_name}.json')

            self.move_files([jpg], os.path.join(self.data_sub_full_path, 'model', 'train', 'images'))
            self.move_files([label_path], os.path.join(self.data_sub_full_path, 'model', 'train', 'labels'))

            percent_complete = ((jpgCount) / totalJpgs) * 100
            print(f"\rProgress: {percent_complete:.2f}%", end='')

        for jpg in evaluteJpgs:
            jpgCount += 1
            img = cv2.imread(jpg)
            img_name = os.path.splitext(os.path.basename(jpg))[0]
            label_path = os.path.join(self.data_sub_full_path, 'model', 'labels', f'{img_name}.json')

            self.move_files([jpg], os.path.join(self.data_sub_full_path, 'model', 'evaluate', 'images'))
            self.move_files([label_path], os.path.join(self.data_sub_full_path, 'model', 'evaluate', 'labels'))

            percent_complete = ((jpgCount) / totalJpgs) * 100
            print(f"\rProgress: {percent_complete:.2f}%", end='')

        for jpg in testJpgs:
            jpgCount += 1
            img = cv2.imread(jpg)
            img_name = os.path.splitext(os.path.basename(jpg))[0]
            label_path = os.path.join(self.data_sub_full_path, 'model', 'labels', f'{img_name}.json')

            self.move_files([jpg], os.path.join(self.data_sub_full_path, 'model', 'test', 'images'))
            self.move_files([label_path], os.path.join(self.data_sub_full_path, 'model', 'test', 'labels'))

            percent_complete = ((jpgCount) / totalJpgs) * 100
            print(f"\rProgress: {percent_complete:.2f}%", end='')

        print("=> Done.")

    def splitListForLoading(self, original_list, percent1, percent2, percent3):
        if percent1 + percent2 + percent3 != 100:
            raise ValueError("The sum of the percentages must equal 100.")

        total_length = len(original_list)
        length1 = int(total_length * percent1 / 100)
        length2 = int(total_length * percent2 / 100)

        # Randomly select elements for the first split
        list1 = random.sample(original_list, length1)

        # Remove the selected elements and randomly select for the second split
        remaining_after_list1 = [item for item in original_list if item not in list1]
        list2 = random.sample(remaining_after_list1, length2)

        # The rest of the elements go into the third list
        list3 = [item for item in remaining_after_list1 if item not in list2]

        return list1, list2, list3
    
    def loadModelIntoDataset(self):
        train_images = tf.data.Dataset.list_files(os.path.join(self.data_sub_full_path, 'model', 'train', 'images', '*.jpg'), shuffle=False)
        train_images = train_images.map(self.loadImage)
        train_images = train_images.map(lambda x: tf.image.resize(x, (240, 240)))
        train_images = train_images.map(lambda x: x / 255)

        test_images = tf.data.Dataset.list_files(os.path.join(self.data_sub_full_path, 'model', 'test', 'images', '*.jpg'), shuffle=False)
        test_images = test_images.map(self.loadImage)
        test_images = test_images.map(lambda x: tf.image.resize(x, (240, 240)))
        test_images = test_images.map(lambda x: x / 255)

        evaluate_images = tf.data.Dataset.list_files(os.path.join(self.data_sub_full_path, 'model', 'evaluate', 'images', '*.jpg'), shuffle=False)
        evaluate_images = evaluate_images.map(self.loadImage)
        evaluate_images = evaluate_images.map(lambda x: tf.image.resize(x, (240, 240)))
        evaluate_images = evaluate_images.map(lambda x: x / 255)

        train_labels = tf.data.Dataset.list_files(os.path.join(self.data_sub_full_path, 'model', 'train', 'labels', '*.json'), shuffle=False)
        train_labels = train_labels.map(lambda x: tf.py_function(self.loadLabels, [x], [tf.uint8, tf.float16]))
        test_labels = tf.data.Dataset.list_files(os.path.join(self.data_sub_full_path, 'model', 'test', 'labels', '*.json'), shuffle=False)
        test_labels = test_labels.map(lambda x: tf.py_function(self.loadLabels, [x], [tf.uint8, tf.float16]))
        evaluate_labels = tf.data.Dataset.list_files(os.path.join(self.data_sub_full_path, 'model', 'evaluate', 'labels', '*.json'), shuffle=False)
        evaluate_labels = evaluate_labels.map(lambda x: tf.py_function(self.loadLabels, [x], [tf.uint8, tf.float16]))

        
        train = tf.data.Dataset.zip((train_images, train_labels))
        train = train.shuffle(4000)
        train = train.batch(8)
        self.train = train.prefetch(4)
        test = tf.data.Dataset.zip((test_images, test_labels))
        test = test.shuffle(1000)
        test = test.batch(8)
        self.test = test.prefetch(4)
        val = tf.data.Dataset.zip((evaluate_images, evaluate_labels))
        val = val.shuffle(1500)
        val = val.batch(8)
        self.val = val.prefetch(4)

        plt.show()

            
    def loadImage(self, x):
        byte_img = tf.io.read_file(x)
        img = tf.io.decode_jpeg(byte_img, channels=3)  # Force RGB channels
        img.set_shape([None, None, 3])  # Set shape to be [height, width, channels]
        return img
    
    def loadLabels(self, label_path):
        with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
            label = json.load(f)

        return [label['class']], label['bbox']

    def getLabelInformation(self, _json, _img, _format='albumentations'):
        img_sizex = _img.shape[1]
        img_sizey = _img.shape[0]

        coords = [0,0,0,0]
        coords[0] = _json['shapes'][0]['points'][0][0]
        coords[1] = _json['shapes'][0]['points'][0][1]
        coords[2] = _json['shapes'][0]['points'][1][0]
        coords[3] = _json['shapes'][0]['points'][1][1]

        reformatted_coords = coords

        if _format == 'albumentations':
            reformatted_coords = list(np.divide(coords, [img_sizex, img_sizey, img_sizex, img_sizey]))

        return reformatted_coords

    def empty_directory(self, directory):
        if not os.path.isdir(directory):
            print(f"{directory} is not a directory or does not exist.")
            return

        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)

            # Remove files and links, skip directories
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)


    def move_files(self, file_paths, destination_folder, delete_originals=False):
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for file_path in file_paths:
            # Ensure the file exists before trying to move it
            if os.path.isfile(file_path):
                destination_path = os.path.join(destination_folder, os.path.basename(file_path))
                
                # Copy the file
                with open(file_path, 'rb') as f_source:
                    with open(destination_path, 'wb') as f_destination:
                        f_destination.write(f_source.read())
                
                # Delete the original file
                if delete_originals:
                    os.remove(file_path)

    def expandInputImages():
        print("** NNI Trainer: Expanding Input Images")

        print("=> Done.")

    def buildModel(self):
        print("** NNI Trainer: Building Model")

        vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(240, 240, 3))

        print (vgg.summary())

        input_layer = tf.keras.layers.Input(shape=(240,240,3))
    
        vgg = tf.keras.applications.VGG16(include_top=False)(input_layer)

        # Classification Model  
        f1 = tf.keras.layers.GlobalMaxPooling2D()(vgg)
        class1 = tf.keras.layers.Dense(2048, activation='relu')(f1)
        class2 = tf.keras.layers.Dense(1, activation='sigmoid')(class1)
        
        # Bounding box model
        f2 = tf.keras.layers.GlobalMaxPooling2D()(vgg)
        regress1 = tf.keras.layers.Dense(2048, activation='relu')(f2)
        regress2 = tf.keras.layers.Dense(4, activation='sigmoid')(regress1)
        
        facetracker = tf.keras.models.Model(inputs=input_layer, outputs=[class2, regress2])
        print("=> Done.")
        return facetracker
    
    def trainModel(self):
        print("** NNI Trainer: Training Model")
        model = self.buildModel()
        X, y = self.train.as_numpy_iterator().next()
        classes, coords = model.predict(X)

        batches_per_epoch = len(self.train)
        lr_decay = (1./0.75 -1)/batches_per_epoch
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, decay=lr_decay)

        classloss = tf.keras.losses.BinaryCrossentropy()
        regressloss = self.localizationLoss

        modelObject = FaceTracker(model)
        modelObject.compile(opt, classloss, regressloss)
        logdir = os.path.join(self.data_sub_full_path,'logs')

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        hist = modelObject.fit(self.train, epochs=10, validation_data=self.val, callbacks=[tensorboard_callback])

        hist.history
        fig, ax = plt.subplots(ncols=3, figsize=(20,5))

        ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
        ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
        ax[0].title.set_text('Loss')
        ax[0].legend()

        ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
        ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
        ax[1].title.set_text('Classification Loss')
        ax[1].legend()

        ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
        ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
        ax[2].title.set_text('Regression Loss')
        ax[2].legend()

        plt.show()

        print("=> Done.")

    def localizationLoss(self, y_true, yhat):
        delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
                  
        h_true = y_true[:,3] - y_true[:,1] 
        w_true = y_true[:,2] - y_true[:,0] 

        h_pred = yhat[:,3] - yhat[:,1] 
        w_pred = yhat[:,2] - yhat[:,0] 
        
        delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
        
        return delta_coord + delta_size

class NeuralNetworkInterface:
    
    def __init__(self):
        print("** NNI: Initializing Neural Network Interface")

        tf.get_logger().setLevel('ERROR')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("=> Done.")
        pass
        
    
