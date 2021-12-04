import torch
import train
import test

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Device: ",device)



print("Training started...")
train.train()

print("Testing started...")
test.test()