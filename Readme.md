# Face Recognition

Before we go deep into this assignment, let us first know what the difference is between face verification and face recognition.

**Face verification** is the process of verifying whether two images belong to the same person or not; it is a 1:1 comparison, 

**Face recognition** is the process of identifying a person from a set of known individuals it is a 1:k comparison. In other words, face verification answers the question "is this person who they claim to be?", while face recognition answers the question "who is this person?".

Face verification is typically used in security systems, such as access control, to ensure that only authorized individuals are granted access.

## 1 - Packages

Import all required packets 


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import PIL

%matplotlib inline
%load_ext autoreload
%autoreload 2
```

## 2. Load the FaceNet model

Pre-trained models have already been trained on large datasets and can perform well on a variety of tasks, including face verification. By using a pre-trained model, you can leverage the knowledge learned by the model in the training phase and fine-tune it on your specific dataset, instead of starting from scratch. This can lead to better performance and faster convergence during training. Additionally, pre-trained models are often made publicly available, making them accessible to researchers and developers who may not have the resources to train their own large models.


```python
model = tf.keras.models.load_model("my_model")
```

    WARNING:tensorflow:SavedModel saved prior to TF 2.5 detected when loading Keras model. Please ensure that you are saving the model with model.save() or tf.keras.models.save_model(), *NOT* tf.saved_model.save(). To confirm, there should be a file named "keras_metadata.pb" in the SavedModel directory.
    WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.
    


```python
FRmodel = model
```

This function will take the image path, resize it to proper dimensions, then normalize it, and then feed it to the model to produce the vector of 128.


```python
def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)
```

It's a good idea to save the embeddings in a database instead of feeding the model with the authorized image of the person each time the user comes. This is because calculating embeddings for an image can be computationally expensive and can take a lot of time


```python
database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)
```

## 3.  Face Verification

This function is used to check if the user in front of the camera is an authorized user or not. The function contains multiple arguments: the image path, user information, the database, and the model. The user image will be converted to embedding, then call the embedding store in the database of the user, compute the norm, check if the difference is less than 0.7, pass otherwise, and prevent the user from entering the system.


```python
def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
        image_path -- path to an image
        identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
        database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
        model -- your Inception model instance in Keras
    
    Returns:
        dist -- distance between the image_path and the image of "identity" in the database.
        door_open -- True, if the door should open. False otherwise.
    """
    ### START CODE HERE
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path, model)
    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding - database[identity])
    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist< 0.7:
        print("It's " + str(identity) + ", welcome in!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
    ### END CODE HERE        
    return dist, door_open
```

### 3.1 Testing 


```python
distance, door_open_flag = verify("images/camera_0.jpg", "younes", database, FRmodel)
```

    It's younes, welcome in!
    


```python
verify("images/camera_2.jpg", "kian", database, FRmodel)
```

    It's not kian, please go away
    




    (1.0259345, False)



## 4. Face Recognition

Implement the `verify()` function, which checks if the front-door camera picture (`image_path`) is actually the person called "identity". You will have to go through the following steps:

- Compute the encoding of the image from `image_path`.
- Compute the distance between this encoding and the encoding of the identity image stored in the database.
- Open the door if the distance is less than 0.7, else do not open it.

As presented above, you should use the L2 distance `np.linalg.norm`.

**Note**: In this implementation, compare the L2 distance, not the square of the L2 distance, to the threshold 0.7.

*Hints*:

- `identity` is a string that is also a key in the database dictionary.
- `img_to_encoding` has two parameters: the image_path and model.


```python
def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
        image_path -- path to an image
        identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
        database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
        model -- your Inception model instance in Keras
    
    Returns:
        dist -- distance between the image_path and the image of "identity" in the database.
        door_open -- True, if the door should open. False otherwise.
    """
    ### START CODE HERE
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path, model)
    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding - database[identity])
    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist< 0.7:
        print("It's " + str(identity) + ", welcome in!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
    ### END CODE HERE        
    return dist, door_open
```

## 4.1 Testing 


```python
verify("images/camera_0.jpg", "younes", database, FRmodel)
```

    It's younes, welcome in!
    




    (0.5992945, True)




```python
verify("images/camera_2.jpg", "kian", database, FRmodel)
```

    It's not kian, please go away
    




    (1.0259345, False)




```python
distance, door_open_flag = verify("images/camera_0.jpg", "younes", database, FRmodel)
```

    It's younes, welcome in!
    


```python

```
