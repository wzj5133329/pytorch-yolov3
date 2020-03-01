# '''
# 将filename1下的文件以绝对路经的形式保存在filename2中
# '''
# import os
# from os import listdir, getcwd
# from os.path import join
# if __name__ == '__main__':

	
#     	filename1 = '/media/asd/DATA/kangni_4/voc_1000_lable'
#     	filename2 = '/media/asd/DATA/kangni_4/voc_1000_lable.txt'
		
#     #source_folder='/media/asd/DATA/PLATE/FAR/'
#     #dest='/home/asd/Project/NCC_WZJ/YOLO/train/PLATE/ccpd_base.txt' 
#     #dest2='/home/asd/Install/yolo-lite_train/train_people/VOCdevkit/VOC2007/ImageSets/Main/val.txt'  
#     	file_list=os.listdir(filename1)       
#     	train_file=open(filename2,'a')                 
#     #val_file=open(dest2,'a')                  
#     	for file_obj in file_list:                
#         	file_path=os.path.join(filename1,file_obj) 
       
#         	file_name,file_extend=os.path.splitext(file_obj)
       
#         #file_num=int(file_name) 
        
#         #if(file_num<8000):                                 
#         	train_file.write(file_path+'\n')  
#         #else :
#         #    val_file.write(file_name+'\n')    
#     	train_file.close()
# #val_file.close()


#将文件夹中的文件名保存在txt中，且简单的按照大比例分，不考虑每一类
import os
from os import listdir, getcwd
from os.path import join
if __name__ == '__main__':

    filename1 = '/home/xiaosa/install/PyTorch-YOLOv3/data/custom/images'
    filename2 = '/home/xiaosa/install/PyTorch-YOLOv3/data/custom/train.txt'  #训练用
    filename3 = '/home/xiaosa/install/PyTorch-YOLOv3/data/custom/valid.txt'    #验证用
    file_list=os.listdir(filename1)       
    train_file=open(filename2,'a')
    val_file=open(filename3,'a')
    print (len(file_list))
    image_num=0
    for file_obj in file_list:                
        file_path=os.path.join(filename1,file_obj) 
		#file_name,file_extend=os.path.splitext(file_obj)
        if(image_num<=len(file_list)*0.8):                                 
            train_file.write(file_path+'\n')
        else:
	        val_file.write(file_path+'\n')
        image_num+=1    
    train_file.close()
    val_file.close()
	#val_file.close()



# #仅仅获取文件名
# import os
# from os import listdir, getcwd
# from os.path import join
# if __name__ == '__main__':

#     filename1 = '/home/xiaosa/data/VOC2012/JPEGImages'
#     filename2 = '/home/xiaosa/install/PyTorch-YOLOv3/data/custom/image_name.txt'  #训练用
#     file_list=os.listdir(filename1)       
#     train_file=open(filename2,'a')
#     print (len(file_list))
#     image_num=0
#     for file_obj in file_list:                
#         file_path=os.path.join(filename1,file_obj) 
# 		#file_name,file_extend=os.path.splitext(file_obj)                           
#         train_file.write(file_path+'\n')
#         image_num+=1    
#     train_file.close()
