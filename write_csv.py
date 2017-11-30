import os
from natsort import natsorted
import pandas as pd


data_path='/mnt/fs3/qa_analitics/Person_Re_Identification/mnist/test'
test=os.listdir(data_path)
namelist=[]
classlist=[]

#namelist_dict = {}

file = open('test_mnist.txt', 'w')

if os.path.isdir(data_path):

    classes=os.listdir(data_path)
    classes = natsorted(classes, key=lambda y: y.lower())
    for class_name in classes:
        images = os.listdir(os.path.join(data_path , class_name))
        images = natsorted(images, key=lambda y: y.lower())
        for image_name in images:
            namelist.append(os.path.join(data_path, class_name, image_name))
            classlist.append(class_name)
            #namelist_dict.update({'path':os.path.join(data_path, class_name, image_name), 'label':class_name})

            file.write(os.path.join(data_path , class_name, image_name)+ ' '+class_name+'\n')
                #print(data_path + name+'/'+class_name+'/'+image_name+'\n')



    file.close()
    df = pd.DataFrame(zip(namelist, classlist), columns=["im_path", "label"])
    df.to_csv('test_mnist.csv', index=False)
