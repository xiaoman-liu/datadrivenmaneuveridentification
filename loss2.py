import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import os
from utils.predict_utils import get_filePath_fileName_fileExt




if __name__ == "__main__":

    # plot the loss and accuracy in training and prediction

    path = "/home/xinjie/xiaoman/experiments/9-layer-64-128-64-3-3-3-3-3-5-5-5-7-29/"
    filename = "FCN_turn_724sample_300epoch.csv"
    epoch = 200


    _,model_name,__ = get_filePath_fileName_fileExt(path + filename)
    data = pd.read_csv(path + filename,nrows = epoch)
    epoch = data.iloc[1:,0]
    loss = data['loss'][1:]
    acc = data['masked_accuracy'][1:]
    val_loss = data['val_loss'][1:]
    val_acc = data['val_masked_accuracy'][1:]


    fig = plt.figure()
    ax1= fig.add_subplot(1,1,1)
    ax2 = ax1.twinx()
    p1 = ax1.plot(epoch,loss,'r-',label = 'train_loss')
    p2 = ax1.plot(epoch,val_loss,'r--',label = 'val_loss')

    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')


    p3 = ax2.plot(epoch,acc,'g-',label = 'train_acc')
    p4 = ax2.plot(epoch,val_acc,'g--',label = 'val_acc')
    ax2.set_ylabel('accuracy')
    plt.title('%s'%model_name)


    fig.legend(loc = 'center right',bbox_to_anchor=(1,0.5), bbox_transform=ax1.transAxes)
    ax1.grid()
    save_path = path +'%s.png'%model_name
    print("picture saved in {}".format(save_path))
    plt.savefig(save_path)
    # plt.show()





