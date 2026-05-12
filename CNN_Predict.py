import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array

def Predict_One_Image(model,Path):
    img = load_img(Path,target_size = (128,128))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array,axis=0)

    prediction = model.predict(img_array)
    prediction_value = prediction[0][0]

    print("Predicted Value : ","Dog" if prediction_value > 0.5 else "Cat")

    plt.imshow(load_img(Path))
    plt.axis(False)
    plt.show()

def get_files(path):
    if not os.path.exists(path):
        raise FileNotFoundError("No such Folder")
    
    files = [file for file in os.listdir(path)]

    return files

def main():
    model = load_model("BEST_DOG_CAT_DETECTION_MODEL.keras")

    for i in range(10):

        Path = os.path.join("Processed_Dataset","Test",np.random.choice(["Cat","Dog"]))

        files = get_files(Path)

        Predict_One_Image(model,os.path.join(Path,files[np.random.randint(0,len(files))]))

if __name__ == "__main__":
    main()