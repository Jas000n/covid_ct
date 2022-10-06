from plot_loss import plot,compare_model
plot("./loss/res50.npy",name="./plot_image/resnet50")
# plot("./loss/vgg19.npy",name="./plot_image/vgg19")
# plot("./loss/res34.npy",name="./plot_image/resnet34")
#compare_model("./loss/res50.npy","./loss/res34.npy","resnet50","resnet34","./plot_image/res34&res50")