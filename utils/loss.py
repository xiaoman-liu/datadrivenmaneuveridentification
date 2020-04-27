import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import os




if __name__ == "_main__":
    filename = 'trainlog/90_train_log.csv'
    _,model_name,__ = get_filePath_fileName_fileExt(filename)
    data = pd.read_csv(filename,nrows = 200)
    epoch = data.iloc[1:,0]
    loss = data['loss'][1:]
    acc = data['acc'][1:]
    # acc = data["masked_accuracy"][1:]
    val_loss = data['val_loss'][1:]
    val_acc = data['val_acc'][1:]
    # val_acc = data['val_masked_accuracy'][1:]

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
    plt.savefig('trainlog/%s.png'%model_name)
    plt.show()




    # plot the box
    # tx0 = 0
    # tx1 = 200
    # #设置想放大区域的横坐标范围
    # ty0 = 0.000
    # ty1 = 3
    # #设置想放大区域的纵坐标范围
    # sx = [tx0,tx1,tx1,tx0,tx0]
    # sy = [ty0,ty0,ty1,ty1,ty0]
    # pl.plot(sx,sy,"purple")
    # axins = inset_axes(ax1, width=1.5, height=1.5, loc='right')
    # #loc是设置小图的放置位置，可以有"lower left,lower right,upper right,upper left,upper #,center,center left,right,center right,lower center,center"
    # axins.plot(x1,y1 , color='red', ls='-')
    # axins.plot(x2,y2 , color='blue', ls='-')
    # axins.axis([0,20000,0.000,0.12])
    # plt.savefig("train_results_loss.png")
    # pl.show()
