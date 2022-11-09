import tensorflow as tf
import pandas as pd
from keras.applications.inception_resnet_v2 import preprocess_input
import requests
import json

filepaths =  'TEST_data/dog.2.jpg'


IMAGE_SIZE  = (224, 224)
test_image = tf.keras.utils.load_img(filepaths
                            ,target_size =IMAGE_SIZE )
test_image = tf.keras.utils.img_to_array(test_image)
test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
test_image = preprocess_input(test_image)

class_names = ['dog','animal','cat','landscape','product','people']


data = json.dumps({"signature_name": "serving_default", "instances": test_image.tolist()})

# send data using POST request and receive prediction result
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/fashion_model:predict', data=data, headers=headers)
prediction = json.loads(json_response.text)['predictions']


df = pd.DataFrame({'pred':prediction[0]})
df = df.sort_values(by='pred', ascending=False, na_position='first')
print(f"## 예측률 : {(df.iloc[0]['pred'])* 100:.2f}%")
print(class_names[df[df == df.iloc[0]].index[0]])