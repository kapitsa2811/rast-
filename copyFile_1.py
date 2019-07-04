import os
from shutil import copy2
import cv2
import pandas as pd
import sys

#basePath="/home/kapitsa/PycharmProjects/cartoon/dataset/rast/"

cwd=os.getcwd()+"//"
'''
    base folder where all images are kept
'''
base="/home/kapitsa/PycharmProjects/cartoon/simpsonRecognition/SimpsonRecognition/"
basePath=base+"//characters//"

'''
    here train-test split will be kept
'''

writePath=base+"//modelData//"

try:
    if not os.path.isdir(writePath):
        os.mkdir(writePath)
except Exception as e:
    print("\n\t e=",e)


dumpPath=base+"/copyData"

try:
    if not os.path.isdir(dumpPath):
        os.mkdir(dumpPath)
except Exception as e:
    print("\n\t e=",e)

'''
    inside base folders these folders indicates cateogary
'''

dirs=["Batman","GreenLantern","Spiderman","WonderWomen"]

indx=0
sourceDirPath=basePath+dirs[indx]
subFolders=os.listdir(sourceDirPath)
#print("\n\t subFolders=",subFolders)

'''
    data frame for reference
'''
df=pd.DataFrame(columns=["index","fileName","label","extension","basePath","subfolder","entirepath"])
#df1=pd.DataFrame(columns=["index","fileName","label","extension","basePath","subfolder","entirepath"])
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

'''
    the below function collects path of all images present at basePath
    and save in dataFrame
'''

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

df=df.sample(frac=1).reset_index(drop=True)
df.to_csv(dumpPath+"//Allfiles.csv",index=False)

'''
    data splitting
    
'''

from sklearn.model_selection import train_test_split

x=df[["index","fileName","extension","basePath","subfolder","entirepath"]]
Y=df[["label"]]
xTrain, xTest, yTrain, yTest = train_test_split(x, Y, test_size = 0.2, random_state = 0)

#print("\n\t xTrain=",xTrain)
print("\n\t xTrain=",xTrain.shape)

xTrain=pd.DataFrame(data=xTrain)
xTest=pd.DataFrame(data=xTest)
yTrain=pd.DataFrame(data=yTrain)
yTest=pd.DataFrame(data=yTest)

xTrain.to_csv(dumpPath+"//train.csv",index=False)
xTest.to_csv(dumpPath+"///test.csv",index=False)
yTrain.to_csv(dumpPath+"//yLabel.csv",index=False)
yTest.to_csv(dumpPath+"//ytest.csv",index=False)
# index	fileName	label	extension	basePath	subfolder	entirepath

for dirName in dirs:

    try:
        if not os.path.isdir(writePath + "train//"):
            os.mkdir(writePath + "train//")
        if not os.path.isdir(writePath+"train//"+dirName):
            os.mkdir(writePath+"train//"+dirName)

        if not os.path.isdir(writePath + "test//"):
            os.mkdir(writePath + "test//")
        if not os.path.isdir(writePath+"test//"+dirName):
            os.mkdir(writePath+"test//"+dirName)
    except Exception as e1:
        print("\n\t exception-",e1)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\n\t line no-", exc_tb.tb_lineno)

        print("\n\t indx=",indx)


'''
    write train files
'''
print("\n\t columns=",xTrain.columns)

expCount=0

'''
    this writes train files to its destination
'''

for (indx,name) in enumerate(xTrain.iterrows()):

    try:
        #print("\n\t indx=",indx,"\n\t name \n:",name)
        #print("\n\t indx=",indx)#,"\n\t name \n:",name)
        imageName=xTrain.iloc[indx,1]
        #imageName=xTrain.loc[indx,"fileName"]
        #path=xTrain.loc[indx,"entirepath"]

        path=xTrain.iloc[indx,5]
        #cl=yTrain.loc[indx,"label"]

        cl=yTrain.iloc[indx,0]
        # print("\n\t imageName=",imageName)
        # print("\n\t label=",cl)
        # print("\n\t path=",path)

        image=cv2.imread(path)

        if not image is None:
            cv2.imwrite(writePath+"//train//"+cl+"//"+imageName,image)
            #input("check")
            imageName=""
    except Exception as e1:

        expCount=expCount+1
        print("\n\t expCount=",expCount)
        #print("\n\t name=",imageName)
        print("\n\t e=",e1)

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\n\t line no-", exc_tb.tb_lineno)

        print("\n\t indx=",indx)
        print("\n\t values=",xTrain.loc[indx,"fileName"])
        input("check")


'''
    write test files
'''

for (indx,name) in enumerate(xTest.iterrows()):

    try:
        #print("\n\t indx=",indx,"\n\t name \n:",name)
        #print("\n\t indx=",indx)#,"\n\t name \n:",name)
        imageName=xTest.iloc[indx,1]
        #imageName=xTrain.loc[indx,"fileName"]
        #path=xTrain.loc[indx,"entirepath"]

        path=xTest.iloc[indx,5]
        #cl=yTrain.loc[indx,"label"]

        cl=yTest.iloc[indx,0]
        # print("\n\t imageName=",imageName)
        # print("\n\t label=",cl)
        # print("\n\t path=",path)

        image=cv2.imread(path)

        #print("\n\t type=",type(image),'\t cl=',cl,"\t imageName=",imageName)

        if not image is None:
            cv2.imwrite(writePath+"//test//"+cl+"//"+imageName,image)
        #input("check")
        imageName=""
    except Exception as e1:

        expCount=expCount+1
        print("\n\t expCount=",expCount)
        #print("\n\t name=",imageName)
        print("\n\t e=",e1)

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\n\t line no-", exc_tb.tb_lineno)

        print("\n\t indx=",indx)
        print("\n\t values=",xTest.iloc[indx,1])
        input("check")

