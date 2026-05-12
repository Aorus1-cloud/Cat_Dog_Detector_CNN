import os
import random
import shutil
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,BatchNormalization,MaxPooling2D,Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import BinaryAccuracy
from sklearn.metrics import confusion_matrix,classification_report

import numpy as np

def Corrupt_file_removal(folder_path):              # During training i got multiple corrupt files and due to which training stopped in btw

    dataset_dir = folder_path  
    removed = 0
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    for root, dirs, files in os.walk(dataset_dir):
        for fname in files:
            fpath = os.path.join(root, fname)

            if not fname.lower().endswith(valid_extensions):
                print(f"Non-image file removed: {fpath}")
                os.remove(fpath)
                removed += 1
                continue

            try:
                with Image.open(fpath) as img:
                    img.load()   
            except Exception as e:
                print(f"Corrupt image removed: {fpath} — {e}")
                os.remove(fpath)
                removed += 1

    print(f"\nDone. Removed {removed} bad files.")

def cat_dog_images(cat,dog):
    if not os.path.isdir(cat):
        raise FileNotFoundError("No Folder of cat")    
    if not os.path.isdir(dog):
        raise FileNotFoundError("No Folder of dog")    
    
    cat_images = []
    dog_images = []

    for cats,dogs in zip(os.listdir(cat),os.listdir(dog)):
        cat_images.append(os.path.join(cat,cats))
        dog_images.append(os.path.join(dog,dogs))
    
    random.shuffle(cat_images)
    random.shuffle(dog_images)

    return cat_images,dog_images

def train_validation_test_split(dataset,class_name,output_folder_name):

    folders = [
        os.path.join(output_folder_name,"Train",class_name),
        os.path.join(output_folder_name,"Validate",class_name),
        os.path.join(output_folder_name,"Test",class_name),
    ]

    for folder in folders:
        os.makedirs(folder,exist_ok = True)

    total_size = len(dataset)

    train_size = int(total_size * 0.70)
    validation_size =  int((total_size - train_size)/2)

    training_images = dataset[:train_size]
    validation_images = dataset[train_size:train_size + validation_size]
    test_images = dataset[train_size + validation_size:]

    seperated_dataset = {
        "Train" : training_images,
        "Validate" : validation_images,
        "Test" : test_images
    }

    for Folder_name,files in seperated_dataset.items():
        DESTINATION_FOLDER = os.path.join(
            output_folder_name,
            Folder_name,
            class_name
        )

        for file in files:
            shutil.copy2(file,os.path.join(DESTINATION_FOLDER,os.path.basename(file)))

def Image_preprocessing(train,validate,test,image_size,batch):
    
    Train_datagen = ImageDataGenerator(
        rescale = 1.0/255,              # Normalize pixel value from 0-255 to 0-1
        rotation_range = 15,            # Randomly rotate image by 15deg
        zoom_range = 0.2,               # Randomly zoom image
        width_shift_range = 0.2,        # Horizontal Shift
        height_shift_range = 0.2,       # Vertical Shift
        horizontal_flip = True,         # Random Horizontal Flip
        shear_range = 0.15              # Shear Transformation
    )

    Validate_datagen = ImageDataGenerator(
        rescale = 1.0/255               # Only Rescaling Bcz its validation dataset and act as testing
    )

    Test_datagen = ImageDataGenerator(
        rescale = 1.0/255               # Only Rescaling Bcz its test dataset
    )

    train_dataset = Train_datagen.flow_from_directory(
        train,
        target_size = (image_size,image_size),
        batch_size = batch,
        class_mode = "binary"
    )

    validation_dataset = Validate_datagen.flow_from_directory(
        validate,
        target_size = (image_size,image_size),
        batch_size = batch,
        class_mode = "binary"
    )

    test_dataset = Test_datagen.flow_from_directory(
        test,
        target_size=(image_size, image_size),
        batch_size=batch,
        class_mode="binary",
        shuffle=False
    )

    return train_dataset,validation_dataset,test_dataset


def main():
    IMAGE_SIZE = 128
    BATCH_SIZE = 32
    EPOCHS = 20
    RANDOM_SEED = 42

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    PET_FOLDER = "PetImages"
    PROCESSED_DATASET = "Processed_Dataset"

    if os.path.exists(PROCESSED_DATASET):
        shutil.rmtree(PROCESSED_DATASET)
    Corrupt_file_removal(PET_FOLDER)

    CAT_PATH = os.path.join(PET_FOLDER,"Cat")
    DOG_PATH = os.path.join(PET_FOLDER,"Dog")

    cat,dog = cat_dog_images(CAT_PATH,DOG_PATH)

    train_validation_test_split(cat,"Cat",PROCESSED_DATASET)
    train_validation_test_split(dog,"Dog",PROCESSED_DATASET)

    train_dataset = os.path.join(PROCESSED_DATASET,"Train")
    validate_dataset = os.path.join(PROCESSED_DATASET,"Validate")
    test_dataset = os.path.join(PROCESSED_DATASET,"Test")

    train_dataset,validate_dataset,test_dataset = Image_preprocessing(train_dataset,validate_dataset,test_dataset,IMAGE_SIZE,BATCH_SIZE)

    print("Class Indices : ",train_dataset.class_indices)

    model = Sequential()

    model.add(Conv2D(32,(3,3),activation = "relu",input_shape = (IMAGE_SIZE,IMAGE_SIZE,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(64,(3,3),activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(128,(3,3),activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(256,(3,3),activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Flatten())

    model.add(Dense(256,activation="relu"))
    model.add(Dropout(0.5))                             # Randomly shuts down 50% of total neurons during training so that no single neuron is dominating and avoids overfitting 

    model.add(Dense(128,activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(1,activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    early_stop = EarlyStopping(                 # Monitors the Loss and if the loss isnt improving(means descreasing) then it stops training
        monitor="val_loss",                     # monitoring parameter
        patience=4,                             # How many epochs to wait before stopping the training
        restore_best_weights=True               # Restores weight back to the best val_loss value
    )

    checkpoint = ModelCheckpoint(               # Saves the model with more Accuracy
        "BEST_DOG_CAT_DETECTION_MODEL.keras",   # Model_file name
        monitor="val_accuracy",                 # parameter to monitor
        save_best_only=True,                    # only to save the best and not others 
        mode="max",                             # higher == better
        verbose=1                               # How much info is printed on the output 
    )

    reduce_lr = ReduceLROnPlateau(              # Reduces the learning rate when model isnt improving
        monitor="val_loss",                     # monitors loss
        factor=0.2,                             # by how much value to reduce the learning rate
        patience=2,                             # how much epochs to wait before updating
        min_lr=0.00001,                         # never go below this value
        verbose=1                               # info printing
    )
    
    model.fit(
        train_dataset,
        validation_data = validate_dataset,
        callbacks = [early_stop,checkpoint,reduce_lr],
        epochs = EPOCHS
    )

    test_loss,test_accuracy = model.evaluate(test_dataset)

    print("Test Loss     :", test_loss)
    print("Test Accuracy :", test_accuracy * 100)

    y_pred = model.predict(test_dataset)

    actual_classes = test_dataset.classes

    accuracy = BinaryAccuracy(threshold=0.5)

    accuracy.update_state(actual_classes,y_pred)

    print("Accuracy of the model on test data is :",accuracy.result())
    print("Confusion Matrix :\n",confusion_matrix(actual_classes,y_pred))
    print("Classification Report :\n",classification_report(actual_classes,y_pred,target_names=test_dataset.class_indices.keys()))

    model.save("BEST_DOG_CAT_DETECTION_MODEL.keras")

    print("Model saved successfully : BEST_DOG_CAT_DETECTION_MODEL.keras")

if __name__ == "__main__":
    main()