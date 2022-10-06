import time
import torch
from torchvision.io import read_image

mymodel = torch.load("/home/jas0n/PycharmProjects/covid_ct/weights/38.pth")
mymodel.to("cuda")
mymodel.eval()
image = read_image("/home/jas0n/PycharmProjects/covid_ct/COVID-CT/all_image_resized/2%1.jpg")
image = image.to("cuda").float().view(1,3,224,224)
time_start = time.time()
result = mymodel(image)
time_end = time.time()
possibility = torch.nn.functional.softmax(result,dim=1)

affected_covid = True if result.argmax().item() else False

print("according to my residual model, ",end="")
if(affected_covid):
    print("the patient is likely to be affected by covid " ,end="")
else:
    print("the patient may not be affected by covid")
print("with a possibility of {}%".format(possibility[0,result.argmax().item()].item()*100))
print("when predicting, using {} seconds".format(time_end-time_start))