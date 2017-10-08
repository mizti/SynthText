# Usage:
# 1. DL following files in data directory
#  http://zeus.robots.ox.ac.uk/textspot/static/db/imnames.cp
#  http://zeus.robots.ox.ac.uk/textspot/static/db/bg_img.tar.gz
#  http://zeus.robots.ox.ac.uk/textspot/static/db/depth.h5
#  http://zeus.robots.ox.ac.uk/textspot/static/db/seg.h5 
# 2. decompress bg_img.tar.gz: now images are in data/bg_img/*.jpg
# 3. Run this with 'python gen_with_fulldata.py'

import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
from synthgen import *
from common import *
import wget, tarfile


## Define some configuration variables:
NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1 # no. of times to use the same image
SECS_PER_IMG = 5 #max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DEPTH_DB_FNAME = osp.join(DATA_PATH,'depth.h5')
SEG_DB_FNAME = osp.join(DATA_PATH,'seg.h5')

VALIDATION_DATA_RATE = 3 # how many images used for validation data chunk

def get_data():
    return h5py.File(DEPTH_DB_FNAME, 'r'), h5py.File(SEG_DB_FNAME, 'r')

def add_res_to_db(imgname,res,db):
    """
    Add the synthetically generated text image instance
    and other metadata to the dataset.
    """
    ninstance = len(res)
    for i in xrange(ninstance):
        dname = "%s_%d"%(imgname, i)
        db['data'].create_dataset(dname,data=res[i]['img'])
        db['data'][dname].attrs['charBB'] = res[i]['charBB']
        db['data'][dname].attrs['wordBB'] = res[i]['wordBB']                
        db['data'][dname].attrs['txt'] = res[i]['txt']

def main(viz=False, validation=False):
    if validation:
        out_file = "results/SynthTextVal.h5"
    else:
        out_file = "results/SynthText.h5"
        
    # open the output h5 file:
    out_db = h5py.File(out_file,'w')
    out_db.create_group('/data')
    print colorize(Color.GREEN,'Storing the output in: '+out_file, bold=True)

    # get the names of the image files in the dataset:
    imnames = []
    with open('data/imnames.cp') as f:
        data = f.read()
        data = data.replace("aV","")
        lines = data.split('\n')
        imnames = filter(lambda x: x.count('jpg'), lines)

    # get depth / seg db
    depth_db, seg_db = get_data()

    N = len(imnames)
    global NUM_IMG
    if NUM_IMG < 0:
        NUM_IMG = N
    start_idx,end_idx = 0,min(NUM_IMG, N)

    RV3 = RendererV3(DATA_PATH,max_time=SECS_PER_IMG)
    for i in xrange(start_idx,end_idx):
        if validation and i % VALIDATION_DATA_RATE != 0:
            continue
        imname = imnames[i]
        try:
            # get the image:
            img = Image.open('data/bg_img/' + imname)

            # get the pre-computed depth:
            #    there are 2 estimates of depth (represented as 2 "channels")
            #    here we are using the second one (in some cases it might be
            #    useful to use the other one):
            depth = depth_db[imname][:].T
            depth = depth[:,:,1]

            # get segmentation:
            seg = seg_db['mask'][imname][:].astype('float32')
            area = seg_db['mask'][imname].attrs['area']
            label = seg_db['mask'][imname].attrs['label']

            # re-size uniformly:
            sz = depth.shape[:2][::-1]
            img = np.array(img.resize(sz,Image.ANTIALIAS))
            seg = np.array(Image.fromarray(seg).resize(sz,Image.NEAREST))

            print colorize(Color.RED,'%d of %d'%(i,end_idx-1), bold=True)
            print(imname)
            res = RV3.render_text(img,depth,seg,area,label,ninstance=INSTANCE_PER_IMAGE,viz=viz)
            if len(res) > 0:
                # non-empty : successful in placing text:
                add_res_to_db(imname,res,out_db)
            # visualize the output:
            if viz:
                if 'q' in raw_input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
                    break
        except:
            traceback.print_exc()
            print colorize(Color.GREEN,'>>>> CONTINUING....', bold=True)
            continue
    seg_db.close()
    depth_db.close()
    out_db.close()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
    parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations')
    #parser.add_argument('--validation', default=False, help='If true, generate small dataset for validation')
    parser.add_argument('--validation', '-v', action="store_true", help='debug mode')
    args = parser.parse_args()
    main(args.viz, args.validation)
