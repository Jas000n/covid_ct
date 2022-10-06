import matplotlib.pyplot as plt
import numpy as np
def plot(npy_path,name="./plot_image/result.png",title="plot"):
    y = np.load(npy_path)
    len_ = len(y)
    y_test = y[0::2]
    y_train = y[1::2]
    x = np.linspace(0,len_/2-1,int(len_/2))
    plt.plot(x,y_train,label="train_loss")
    plt.plot(x,y_test,label="test_loss")
    plt.legend()
    plt.savefig(name)
    plt.show()

#input path of npy and names, generate comparison between 2 model
def compare_model(npy_path1,npy_path2,model_name1,model_name2,name="./plot_image/compare_result.png",):
    y1 = np.load(npy_path1)
    y2 = np.load(npy_path2)
    len_ = len(y1)
    y1_test = y1[0::2]
    y1_train = y1[1::2]
    y2_test = y2[0::2]
    y2_train = y2[1::2]
    x = np.linspace(0, len_ / 2 - 1, int(len_ / 2))
    plt.plot(x, y1_train, label=model_name1+"train_loss")
    #plt.plot(x, y1_test, label=model_name1+"test_loss")
    plt.plot(x, y2_train, label=model_name2 + "train_loss")
    #plt.plot(x, y2_test, label=model_name2 + "test_loss")
    plt.legend()
    plt.savefig(name)
    plt.show()