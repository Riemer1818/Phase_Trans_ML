import pickle

f = './train_data/normal_2D_20grid_1000itir_100step/rawdata1.pkl'
with open(f, "rb") as file:
    totdata = pickle.load(file)
    dims = totdata[0]
    n = totdata[1]
    dataset = totdata[2]

# # just show a bunch of states
# i=0
# while i < 10:
#     for state in dataset[5]:
#         print(state[0], state[1])
#         i += 1

x = 90 #stats at 90
# # pick a specific state
print(dataset[x][0], dataset[x][1])
print(dataset[x+8][0], dataset[x+8][1])

# subtract two states
print(dataset[x][1] - dataset[x+8][1])