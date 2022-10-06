import torch
import pandas as pd
import numpy as np
# load model
from torchvision.io import read_image

from dataset import Covid_dataset
tosave = []
for j in range(0,50):
        print(j)
        TP, FP, TN, FN = 0, 0, 0, 0
        precision,recall = 0,0
        F1=0
        mymodel = torch.load("/home/jas0n/PycharmProjects/covid_ct/weights/vgg/"+str(j)+".pth")
        mymodel.to("cuda")

        # load dataset
        negative = pd.read_csv("/home/jas0n/PycharmProjects/covid_ct/COVID-CT/Data-split/non_covid.csv")["image_name"].tolist()
        positive = pd.read_csv("/home/jas0n/PycharmProjects/covid_ct/COVID-CT/Data-split/covid.csv")["image_name"].tolist()
        prefix = "/home/jas0n/PycharmProjects/covid_ct/COVID-CT/all_image_resized"
        for i in range(len(negative)):
                image = read_image(prefix+"/"+negative[i])
                image = image.to("cuda").float().view(1,3,224,224)
                result = mymodel(image).argmax().item()
                if(result==0):
                        TN+=1
                else:
                        FP+=1
        print("TN = ",TN)
        print("FP = ",FP)
        for i in range(len(positive)):
                image = read_image(prefix+"/"+positive[i])
                image = image.to("cuda").float().view(1,3,224,224)
                result = mymodel(image).argmax().item()
                if(result==1):
                        TP+=1
                else:
                        FN+=1
        print("TP = ",TP)
        print("FN = ",FN)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1 = 2*precision*recall/(precision+recall)
        print("precision=",precision)
        print("recall=",recall)
        print("F1 score = ",F1)
        tosave.append(TP)
        tosave.append(TN)
        tosave.append(FP)
        tosave.append(FN)
        tosave.append(precision)
        tosave.append(recall)
        tosave.append(F1)
np.save("./precision_recall/vgg19.npy",tosave)
