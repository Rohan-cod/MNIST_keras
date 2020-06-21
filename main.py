from keras.models import model_from_json


json_file = open("model.json", 'r')

model_json = json_file.read()

json_file.close()

model = model_from_json(model_json)