This is the backend for all 'data' used in the framework, which is handled by the Kettle objects.
To implement a new dataset note that ```datasets.py``` contains implementations that inherit from the usual torchvision dataloaders, but overwrite the ```__getitem__``` method to also return the image id and add a ```get_target``` that just returns the id and the target label. The training loops in the ```victims``` module expect this behavior. 


