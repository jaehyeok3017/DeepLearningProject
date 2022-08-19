import os 
import shutil 

def fileList(path_before : str)->list :
    file_list = os.listdir(path_before)
    category = [] 
    for file in file_list:
        temp_list = file.split("_") 
        category.append(temp_list[0]) 

    temp_set = set(category)
    result = list(temp_set) 
    return result


def makeFolder(path_after : str, file_list : list):    
    for file in file_list:
        try:
            os.makedirs(path_after+"/"+file)
        except:
            pass


def moveFile(path_before, path_after):
    folderlist = os.listdir(path_after) 
    filelist = os.listdir(path_before)
    dict = {} 

    for file in filelist:
        temp_list = file.split("_") 
        dict[file] = temp_list[0]
    
    for key, value in dict.items():
        shutil.move(path_before+"/"+key, path_after+"/"+value)


#분류할 파일이 있는 위치 폴더
path_before = r"C:\\Users\\user\\Desktop\\AI\\DeepLearningProject\\model\\src"
file_list = fileList(path_before)

#옮길 경로 폴더
path_after = r"C:\\Users\\user\\Desktop\AI\\DeepLearningProject\\model\\data"
makeFolder(path_after, file_list)
moveFile(path_before, path_after)