import torch
for i in range(1,10):
    path_c = 'c' + str(i) + '.pt'
    c = torch.load(path_c)
    print(i)
    print(c)
    print('-------------')