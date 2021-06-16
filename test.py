import os
import pickle

data = []
number_of_data = 0 
directory = "./train_data5/normal_2D_20grid_10itir_10step"

for filename in os.listdir(directory):

    if filename.endswith('.pkl'):
        
        f = os.path.join(directory, filename)
        with open(f, "rb") as file:
            dataset = pickle.load(file)
            print("here")
            number_of_data += 1
        
        data.append(dataset)

    else:
        pass

print(data)