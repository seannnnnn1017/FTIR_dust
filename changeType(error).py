import os
import os.path as osp

import pandas as pd

def change_DataType(count, list_data):
    root_path = data_root_path
    for out_name in list_data:
        now_file = root_path + out_name
        checkfile = os.listdir(now_file)
        for inside_name in checkfile:
            now_data = now_file+'/'+inside_name 
            checkdata = os.listdir(now_data)
            #print(checkdata)
            for i in range(len(checkdata)):
                if '.asc' in checkdata[i]:
                    print(checkdata[i])
                    name_split = checkdata[i].split('.')
                    newpath = name_split[0]+".txt"
                    # newroot = out_name + inside_name
                    newroot = inside_name+"_newFile"
                    path = "data_wave/"+out_name+"/"+newroot   
                    isExist = os.path.exists(path)
                    
                    if isExist == True:
                        pass    
                    else:
                        os.mkdir(path)
                    
                    print(f'checkdata : {checkdata[i]}')
                    print(f'newpath : {newpath}')
                    print(f'newroot : {newroot}')
                    print('-'*10)
                    raw_file = "data_wave/" + out_name +"/"+ inside_name+"/"
                    file = path+"/"+name_split[0]+".txt"
                    os.rename(raw_file+checkdata[i], file)
                    
                    #modify the file content
                    file_path = file
                    files = open(file_path)
                    lines = files.readlines()
                    
                    del lines[0: 56]
                    files.close()

                    file_new = open(file_path, 'w+')
                    file_new.writelines(lines)
                    file_new.close()
                    

                    
                    # name_split = checkdata[i].split('.')
                    # print(name_split[0])
                    # file_asc = name_split[0]+'.txt'`  
                    # pf = pd.read_csv(file_asc)
                    # print(pf)
                    # print(i,'-'*100)
                    
                    



data_root_path = 'data_wave/'
# wavedata = data_root_path + 'data_wave'
datalist = os.listdir(data_root_path)
print(datalist)
datalist.pop(0)
change_DataType(len(datalist), datalist)
