import os
from shutil import copy2
import cv2
import pandas as pd


basePath="/home/kapitsa/PycharmProjects/cartoon/dataset/rast/"
dirs=["Batman","GreenLantern","Spiderman","WonderWoman"]
dirs=["Batman","GreenLantern","Spiderman","WonderWoman"]
indx=1
sourceDirPath=basePath+dirs[indx]
subFolders=os.listdir(sourceDirPath)
print("\n\t subFolders=",subFolders)

df=pd.DataFrame(columns=["index","fileName","label","extension","basePath","subfolder","entirepath"])
df1=pd.DataFrame(columns=["index","fileName","label","extension","basePath","subfolder","entirepath"])
count=0


'''
for indx,subDirName in enumerate(subFolders):

    for i,nm in enumerate(os.listdir(sourceDirPath+subDirName)):
        print("\n\t subFolder=",subDirName,"\t name=",nm)
        fileName=os.path.join(sourceDirPath,subDirName,nm)
        print("\t full path=",os.path.isfile(fileName))
        count=count+1

#print("\n\t count=",count)
'''



import os.path




def absoluteFilePaths(directory,exetensions,d,totCount):
    count=0
    for dirpath,_,filenames in os.walk(directory):
       for f in filenames:

           f1=f.lower()
           if f1.endswith(".jpg") or f1.endswith(".jpeg") or f1.endswith(".png") :#or f1.endswith(".gif")
               #yield os.path.abspath(os.path.join(dirpath, f))
               nm=os.path.abspath(os.path.join(dirpath, f))
               #df.append(df,[count,f,basePath,dirpath,nm])
               ext = os.path.splitext(f)[1]
               df.loc[totCount+count]=[count,f,d,ext,basePath,dirpath,nm]

               exetensions.add(ext)
               #exetensions.__add__(ext)
               count=count+1

    return df,count,exetensions

totCount=0
exetensions=set()
for indx,d in enumerate(dirs):
    sourceDirPath = basePath + dirs[indx]
    df,count,exetensions=absoluteFilePaths(sourceDirPath,exetensions,d,totCount)
    totCount=totCount+count
    print("\n\t dir=",d,"\t count=", count,"\t totCount=",totCount)

print("\n\t totCount=",totCount)
#print("\n\t df=\n",df.head())
print("\n\t ext=",exetensions)
df.to_csv("/home/kapitsa/PycharmProjects/cartoon//Allfiles.csv")

'''
    data splitting
    
'''

from sklearn.model_selection import train_test_split

x=df[["index","fileName","extension","basePath","subfolder","entirepath"]]
Y=df[["label"]]
xTrain, xTest, yTrain, yTest = train_test_split(x, Y, test_size = 0.2, random_state = 0)

print("\n\t xTrain=",xTrain)
print("\n\t xTrain=",xTrain.shape)

xTrain=pd.DataFrame(data=xTrain)
xTest=pd.DataFrame(data=xTest)
yTrain=pd.DataFrame(data=yTrain)
yTest=pd.DataFrame(data=yTest)

xTrain.to_csv("/home/kapitsa/PycharmProjects/cartoon//train.csv")
xTest.to_csv("/home/kapitsa/PycharmProjects/cartoon//test.csv")
yTrain.to_csv("/home/kapitsa/PycharmProjects/cartoon//yLabel.csv")
yTest.to_csv("/home/kapitsa/PycharmProjects/cartoon//ytest.csv")
# index	fileName	label	extension	basePath	subfolder	entirepath

writePath="/home/kapitsa/PycharmProjects/cartoon/modelData//"
for dirName in dirs:

    try:
        os.mkdir(writePath+"train//"+dirName)
        os.mkdir(writePath+"test//"+dirName)
    except Exception as e:
        print("\n\t exception-",e)

'''
    write train files
'''
print("\n\t columns=",xTrain.columns)

expCount=0
for (indx,name) in enumerate(xTrain.iterrows()):

    try:
        #print("\n\t indx=",indx,"\n\t name \n:",name)
        #print("\n\t indx=",indx,"\n\t name \n:",name[4])
        imageName=xTrain.loc[indx,"fileName"]
        path=xTrain.loc[indx,"entirepath"]
        cl=yTrain.loc[indx,"label"]
        print("\n\t imageName=",imageName)
        print("\n\t label=",cl)
        print("\n\t path=",path)

        image=cv2.imread(path)
        cv2.imwrite(writePath+"//train//"+cl+"//"+imageName,image)
        #input("check")
    except Exception as e:
        import sys
        expCount=expCount+1
        print("\n\t expCount=",expCount)
        print("\n\t name=",imageName)
        print("\n\t e=",e)

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\n\t line no-", exc_tb.tb_lineno)

'''
    write test files
'''