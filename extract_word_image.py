import os
import os.path as osp
import numpy as np
import h5py
from common import *
import matplotlib.pyplot as plt

def flatten(txt):
    def spl(t):
        return t.split("\n")
    flatten = lambda l: [item for sublist in l for item in sublist]
    
    txt = map(spl, txt)
    txt = flatten(txt)
    txt = filter(lambda x: x!="", txt)
    return txt
    
def main(db_fname):
    with h5py.File(db_fname, 'r') as db:
        dsets = sorted(db['data'].keys())
        for k in dsets:
            print(k)
            rgb = db['data'][k][...]
            wordBB = db['data'][k].attrs['wordBB']
            txt = db['data'][k].attrs['txt']
            txt = flatten(txt)
            print(txt)

            plt.close(1)
            plt.figure(1)
            plt.imshow(rgb)
            plt.hold(True)
            H,W = rgb.shape[:2]
            print(txt)
            for i in xrange(wordBB.shape[-1]):
                bb = wordBB[:,:,i]
                bb = np.c_[bb,bb[:,0]]
                print(bb)
                vcol = ["r", "g", "b", "k"]
                # 0=x, 1=y
                # 0 left top
                # 1 right top
                # 2 right bottom
                # 3 left bottom
                print(txt[i])
                for j in xrange(4):
                    plt.scatter(bb[0,j],bb[1,j],color=vcol[j])
                plt.gca().set_xlim([max(bb[0,0],bb[0,3]), min(bb[0,1],bb[0,2])])
                plt.gca().set_ylim([max(bb[1,2],bb[1,3]), min(bb[1,0],bb[1,1])])
                plt.show(block=False)
                raw_input("wait")

            print(txt) # list
            for i in enumerate(txt):
                print(i)
            #viz_textbb(rgb, [charBB], wordBB)

if __name__=='__main__':
    main('results/SynthText.h5')
