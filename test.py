import os
import pickle
import numpy as np

data = []
number_of_data = 0 
#directory = "C:\\Users\\karel\\Documents\\UCU\\SEM8\\Complex_Systems_Project\\Data\\test_data_ML\\normal_2D_20grid_30itir_100step"

for filename in os.listdir(directory):

    if filename.endswith('.pkl'):
        
        f = os.path.join(directory, filename)
        with open(f, "rb") as file:
            dataset = pickle.load(file)
            print(number_of_data)
            number_of_data += 1
            for i in range(len(dataset)):
                data.append(dataset[i])
    else:
        pass

data = np.array(data)
print(data[0])

#%%

#np.save(os.path.join(directory,'data_normal_2D_20grid_30itir_100step'),data)
