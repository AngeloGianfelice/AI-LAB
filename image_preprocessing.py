from PIL import Image
import os
import splitfolders 

raw_data_path='D://Github//AI-LAB//data//raw_data//'
processed_data_path='D://Github//AI-LAB//data//processed_data//'


def Convert_to_RGB(path):
    for i,image in enumerate(os.listdir(path)):
        im = Image.open(path+image)
        if im.mode != 'RGB':
            print("image converted: ",image)
            im.convert("RGB").save(path + f"RGB_image{i}.jpg")
            os.remove(path+image)

if __name__ =='__main__':
    #Convert all images into RGB form
    Convert_to_RGB(raw_data_path + 'Class0//')
    Convert_to_RGB(raw_data_path + 'Class1//')
    #perform train,validation,test split
    splitfolders.fixed(input=raw_data_path,output=processed_data_path,fixed=(1500,250,250))
