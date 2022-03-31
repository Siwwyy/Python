'''
    Own includes
'''

from config import current_device
from Data_Set import CIFAR100_DS
from Image_DataLoader import EAvailable_Datasets, Image_DataLoader

########
import torch
import torchvision
from torchvision.transforms import ToTensor






















































#train_loader = Image_DataLoader(CIFAR100_DS(transforms=ToTensor()))
#test_loader = Image_DataLoader(CIFAR100_DS(train=False,
#transforms=ToTensor()))


#class MyDataset(torch.utils.data.Dataset):
#    def __init__(self):
#        self.data = torch.arange(100).view(100, 1).float()
        
#    def __getitem__(self, index):
#        print(index)
#        x = self.data[index]
#        return x
    
#    def __len__(self):
#        return len(self.data)

#dataset = MyDataset()
#sampler = torch.utils.data.sampler.BatchSampler(
#    torch.utils.data.sampler.RandomSampler(dataset),
#    batch_size=10,
#    drop_last=False)

#loader = torch.utils.data.DataLoader(dataset, batch_size=1)

#for data in loader:
#    print(data)











#train_ds, test_ds = CIFAR100_DS().get_ds()

#print(len(train_ds))



#ssl._create_default_https_context = ssl._create_unverified_context

#transform =
#torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
#print(type(transform))
#temp = torchvision.datasets.CIFAR100(root="data/",
#                                     train=True,
#                                     transform=transform,
#                                     download=True)
#temp1 = torchvision.datasets.CIFAR100(root="data/",
#                                     train=False,
#                                     transform=transform,
#                                     download=True)

#print(temp)









#from collections.abc import Callable
#from typing import Optional, Union

#def foo(x, y:Optional[int]) -> int:
#    #return x * y
#    print(type(x))
#    print(type(y))

#print(foo(10))

#def my_inc(x: Optional[int]=None) -> int:
#    if x is None:
#        return 0
#    else:
#        # The inferred type of x is just int here.
#        return x + 1

#print(my_inc(1))




#def foo(x, y:Union[str]=None) -> int:
#    #return x * y
#    print(type(x))
#    print(type(y))


#print(foo(1, 1.0))


#def fib(n:int):
#    a, b = 0, 1
#    for _ in range(n):
#        yield a
#        b, a = a + b, b

#print([n for n in fib(3.6)])

#from typing import List, Tuple, Union, Any

#l2: List[Union[int]] = ['text', 1, 2, True]
#l1 = ['text', 1, 2, 3]
#print(l1)
#print(type(l2[0]))
#from typing import Mapping, Optional, Sequence, Union
#from collections import OrderedDict

#def test(a: Optional[Mapping[str, int]]=None) -> None:
#    print(a)

#d = dict()
#test(10)


#print(torch.ones(10,5)[0])


from Interpolation import nearest_neighbor_interpolation, linear_interpolation





#print(nearest_neighbor_interpolation())

tens = torch.tensor([[40, 10], [40, 10]], dtype=torch.float32)
print(linear_interpolation(tens, (2, 6)))
##print(torch.nn.functional.interpolate(tens.view(1,1,2), size=4, mode='linear'))
#print(linear_interpolation())


#tens = torch.tensor([10,20], dtype=torch.float32)
#print(linear_interpolation(tens, (4,)))
#tens_start = torch.tensor([10], dtype=torch.float32)
#tens_end = torch.tensor([40], dtype=torch.float32)
##print(torch.lerp(tens_start, tens_end, 0.25))
#sz = (4,)
#print(tens.shape)
#print(sz)
#print(torch.nn.functional.interpolate(tens.view(1,1,2), size=4, mode='linear'))