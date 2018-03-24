from keras.models import Model
from keras.models import load_model
import numpy as np
from PIL import Image
import os
import random
from scipy.misc import imread
import operator
# define input image preprocess 
def preprocess_image (x):
	x /= 256.
	x -= 0.5
	#x *= 2.
	return x


# define product cutting top fully-connected layer
def produce_notop_weight(weight_path, n):
    model = load_model(weight_path)
    model = Model(inputs=model.input, outputs=model.layers[-n].output)
    output_path = weight_path[:-5] + "_notop.hdf5"
    model.save(output_path)

def tri_pic_queue_generator(pair_path):
    file_ = open(pair_path, "r")
    line_sum = 0
    anchor_list = []
    pos_list = []
    neg_list = []
    for line in file_:
        line_sum += 1
    anchor_list = np.zeros((line_sum, 80, 80, 3))
    pos_list = np.zeros((line_sum, 80, 80, 3))
    neg_list = np.zeros((line_sum, 80, 80, 3))
    
    cnt = 0
    file_2 = open(pair_path, "r")
    for line in file_2:
        line = line.rstrip('\n')
        list_ = line.split(" ")
        
        img_list = []
        img_temp = []
        for i in range(len(list_)):
             hw_tuple = (80, 80)
             temp = Image.open(list_[i])
             temp = temp.convert("RGB")
             temp = temp.resize(hw_tuple)
             img_list.append(temp)

        anchor_list[cnt, :, :, :] = preprocess_image(np.array(img_list[0]).astype(float))
        pos_list[cnt, :, :, :] = preprocess_image(np.array(img_list[1]).astype(float))
        neg_list[cnt, :, :, :] = preprocess_image(np.array(img_list[2]).astype(float))
        cnt += 1  
    return anchor_list, pos_list, neg_list

def tri_image_epoch_generator(image_dir_path, num_of_pics_in_epoch, image_size):
    dict_ = {}
    set_ = set()
    for folder in os.listdir(image_dir_path):
        folder_path = image_dir_path + '/' + folder
        for file_ in os.listdir(folder_path):
            if folder_path not in dict_:
                dict_[folder_path] = []
            file_path = folder_path + '/' + file_
            dict_[folder_path].append(file_path)

    folder_list = list(dict_.keys()) 
    
    anchor_list = np.zeros((num_of_pics_in_epoch, image_size, image_size, 3))
    pos_list = np.zeros((num_of_pics_in_epoch, image_size, image_size, 3))
    neg_list = np.zeros((num_of_pics_in_epoch, image_size, image_size, 3))
    cnt = 0
    while cnt < num_of_pics_in_epoch:

        anchor_folder_index = 0
        neg_folder_index = 0
        while anchor_folder_index == neg_folder_index:
            anchor_folder_index = random.randint(0, len(folder_list) - 1)
            neg_folder_index = random.randint(0, len(folder_list) - 1)
        
        anchor_file_list = dict_[folder_list[anchor_folder_index]]
        anchor_index = 0
        pos_index = 0
        while anchor_index == pos_index:
            anchor_index = random.randint(0, len(anchor_file_list) - 1)
            pos_index = random.randint(0, len(anchor_file_list) - 1)
        
        neg_file_list = dict_[folder_list[neg_folder_index]]
        neg_index = random.randint(0, len(neg_file_list) - 1)

        anchor_file = anchor_file_list[anchor_index]
        pos_file = anchor_file_list[pos_index]
        neg_file = neg_file_list[neg_index]
        tri_tuple = (anchor_file, pos_file, neg_file)
        if tri_tuple not in set_:
            set_.add(tri_tuple)
        else:
            #print "this tri_image has already exits"
            #print tri_tuple
            #print "tri_image has been produced: " + str(cnt)
            continue
        hw_tuple = (image_size, image_size)
        file_list = [anchor_file, pos_file, neg_file]
        img_list = []
        for file_ in file_list:
            temp = Image.open(file_)
            temp = temp.convert("RGB")
            temp = temp.resize(hw_tuple)
            img_list.append(temp)
        

        anchor_list[cnt, :, :, :] = preprocess_image(np.array(img_list[0]).astype(float))
        pos_list[cnt, :, :, :] = preprocess_image(np.array(img_list[1]).astype(float))
        neg_list[cnt, :, :, :] = preprocess_image(np.array(img_list[2]).astype(float))

        cnt += 1
    
    return anchor_list, pos_list, neg_list


def tri_image_epoch_generator_with_label(image_dir_path, num_of_pics_in_epoch, image_size):
    dict_ = {}
    set_ = set()
    for folder in sorted(os.listdir(image_dir_path)):
        folder_path = image_dir_path + '/' + folder
        for file_ in os.listdir(folder_path):
            if folder_path not in dict_:
                dict_[folder_path] = []
            file_path = folder_path + '/' + file_
            dict_[folder_path].append(file_path)

    folder_list = sorted(list(dict_.keys())) 
    
    anchor_list = np.zeros((num_of_pics_in_epoch, image_size, image_size, 3))
    pos_list = np.zeros((num_of_pics_in_epoch, image_size, image_size, 3))
    neg_list = np.zeros((num_of_pics_in_epoch, image_size, image_size, 3))
    cnt = 0
    anchor_label = []
    pos_label = []
    neg_label = []

    while cnt < num_of_pics_in_epoch:

        anchor_folder_index = 0
        neg_folder_index = 0
        while anchor_folder_index == neg_folder_index:
            anchor_folder_index = random.randint(0, len(folder_list) - 1)
            neg_folder_index = random.randint(0, len(folder_list) - 1)
        
        anchor_file_list = dict_[folder_list[anchor_folder_index]]
        anchor_index = 0
        pos_index = 0
        while anchor_index == pos_index:
            anchor_index = random.randint(0, len(anchor_file_list) - 1)
            pos_index = random.randint(0, len(anchor_file_list) - 1)
        
        neg_file_list = dict_[folder_list[neg_folder_index]]
        neg_index = random.randint(0, len(neg_file_list) - 1)

        anchor_file = anchor_file_list[anchor_index]
        pos_file = anchor_file_list[pos_index]
        neg_file = neg_file_list[neg_index]
        tri_tuple = (anchor_file, pos_file, neg_file)
        if tri_tuple not in set_:
            set_.add(tri_tuple)
        else:
            #print "this tri_image has already exits"
            #print tri_tuple
            #print "tri_image has been produced: " + str(cnt)
            continue
        hw_tuple = (image_size, image_size)
        file_list = [anchor_file, pos_file, neg_file]
        img_list = []
        for file_ in file_list:
            try:
                temp = Image.open(file_)
                temp = temp.convert("RGB")
                temp = temp.resize(hw_tuple)
                img_list.append(temp)
            except BaseException as err:
                continue
       
        if (len(img_list)==3): 
            anchor_label.append(anchor_folder_index)
            pos_label.append(anchor_folder_index)
            neg_label.append(neg_folder_index)

            anchor_list[cnt, :, :, :] = preprocess_image(np.array(img_list[0]).astype(float))
            pos_list[cnt, :, :, :] = preprocess_image(np.array(img_list[1]).astype(float))
            neg_list[cnt, :, :, :] = preprocess_image(np.array(img_list[2]).astype(float))

            cnt += 1
    
    return anchor_list, pos_list, neg_list, np.asarray(anchor_label),np.asarray(pos_label),np.asarray(neg_label)

def pair_image_epoch_generator_new(image_dir_path, image_size):
    dict_ = {}
    set_ = set()
    for folder in os.listdir(image_dir_path):
        folder_path = image_dir_path + '/' + folder
        for file_ in os.listdir(folder_path):
            if folder_path not in dict_:
                dict_[folder_path] = []
            file_path = folder_path + '/' + file_
            dict_[folder_path].append(file_path)

    folder_list = list(dict_.keys())
    #print folder_list
    '''
    cnt = 0
    for folder_path in folder_list:
        file_list = dict_[folder_path]
        len_ = len(file_list)
        cnt += len_ * (len_ - 1) / 2
    
    anchor_list = np.zeros((cnt, image_size, image_size, 3))
    pos_list    = np.zeros((cnt, image_size, image_size, 3))
    neg_list    = np.zeros((cnt, image_size, image_size, 3))
    '''
    anchor_pos_neg_list = []
    while (len(anchor_pos_neg_list) < 1000):
        anchor_folder_index = 0
        neg_folder_index = 0
        while anchor_folder_index == neg_folder_index:
            anchor_folder_index = random.randint(0, len(folder_list) - 1)
            neg_folder_index = random.randint(0, len(folder_list) - 1)
        
        anchor_file_list = dict_[folder_list[anchor_folder_index]]
        anchor_index = 0
        pos_index = 0
        while anchor_index == pos_index:
            anchor_index = random.randint(0, len(anchor_file_list) - 1)
            pos_index = random.randint(0, len(anchor_file_list) - 1)
        
        neg_file_list = dict_[folder_list[neg_folder_index]]
        neg_index = random.randint(0, len(neg_file_list) - 1)

        anchor_file = anchor_file_list[anchor_index]
        pos_file = anchor_file_list[pos_index]
        neg_file = neg_file_list[neg_index]
        tri_tuple = (anchor_file, pos_file, neg_file)
        if tri_tuple not in set_:
            set_.add(tri_tuple)
            anchor_pos_neg_list.append([anchor_file,pos_file,neg_file])
        else: 
            continue            
        
    size = len(anchor_pos_neg_list)
    print(size)
    anchor_list = np.zeros((size, image_size, image_size, 3))
    pos_list    = np.zeros((size, image_size, image_size, 3))
    neg_list    = np.zeros((size, image_size, image_size, 3))

    hw_tuple = (image_size, image_size)
    
    cnt = 0
    for tri_image in anchor_pos_neg_list:
        img_list = []
        for file_ in tri_image:
            temp = Image.open(file_)
            temp = temp.convert("RGB")
            temp = temp.resize(hw_tuple)
            img_list.append(temp)
        anchor_list[cnt, :, :, :] = preprocess_image(np.array(img_list[0]).astype(float))
        pos_list[cnt, :, :, :] = preprocess_image(np.array(img_list[1]).astype(float))
        neg_list[cnt, :, :, :] = preprocess_image(np.array(img_list[2]).astype(float))
        cnt += 1
   
    
    return anchor_list, pos_list, neg_list   

def pair_image_epoch_generator_new_with_label(image_dir_path, image_size):
    dict_ = {}
    set_ = set()
    for folder in sorted(os.listdir(image_dir_path)):
        folder_path = image_dir_path + '/' + folder
        for file_ in os.listdir(folder_path):
            if folder_path not in dict_:
                dict_[folder_path] = []
            file_path = folder_path + '/' + file_
            dict_[folder_path].append(file_path)

    folder_list = sorted(list(dict_.keys()))
    #print folder_list
    '''
    cnt = 0
    for folder_path in folder_list:
        file_list = dict_[folder_path]
        len_ = len(file_list)
        cnt += len_ * (len_ - 1) / 2
    
    anchor_list = np.zeros((cnt, image_size, image_size, 3))
    pos_list    = np.zeros((cnt, image_size, image_size, 3))
    neg_list    = np.zeros((cnt, image_size, image_size, 3))
    '''
    anchor_pos_neg_list = []
    set_ = set()
    for k in range(len(folder_list)):
        file_list = dict_[folder_list[k]]
        for i in range(len(file_list) - 1):
            for j in range(i + 1, len(file_list)):
                m = k
                while(m == k):
                    m = random.randint(0, len(folder_list) - 1)
                    neg_file_list = dict_[folder_list[m]]
                    n = random.randint(0, len(neg_file_list) - 1)
                    neg_file = neg_file_list[n]
                    if (file_list[i], neg_file) in set_:
                        continue
                    else:
                        set_.add((file_list[i], neg_file))
                anchor_pos_neg_list.append([file_list[i], file_list[j], neg_file])

    #for ele in anchor_pos_neg_list:
    #    print ele
    size = len(anchor_pos_neg_list)
    anchor_list = np.zeros((size, image_size, image_size, 3))
    pos_list    = np.zeros((size, image_size, image_size, 3))
    neg_list    = np.zeros((size, image_size, image_size, 3))

    hw_tuple = (image_size, image_size)
    
    cnt = 0
    for tri_image in anchor_pos_neg_list:
        img_list = []
        for file_ in tri_image:
            temp = Image.open(file_)
            temp = temp.convert("RGB")
            temp = temp.resize(hw_tuple)
            img_list.append(temp)
        anchor_list[cnt, :, :, :] = preprocess_image(np.array(img_list[0]).astype(float))
        pos_list[cnt, :, :, :] = preprocess_image(np.array(img_list[1]).astype(float))
        neg_list[cnt, :, :, :] = preprocess_image(np.array(img_list[2]).astype(float))
        cnt += 1
   
    
    return anchor_list, pos_list, neg_list   



def pair_image_epoch_generator(image_dir_path, num_of_pics_of_same_person, image_size):
    dict_ = {}
    set_ = set()
    for folder in os.listdir(image_dir_path):
        folder_path = image_dir_path + '/' + folder
        for file_ in os.listdir(folder_path):
            if folder_path not in dict_:
                dict_[folder_path] = []
            file_path = folder_path + '/' + file_
            dict_[folder_path].append(file_path)

    folder_list = list(dict_.keys())
    #print folder_list
    anchor_list = np.zeros((num_of_pics_of_same_person, image_size, image_size, 3))
    pos_list    = np.zeros((num_of_pics_of_same_person, image_size, image_size, 3))
    neg_list    = np.zeros((num_of_pics_of_same_person, image_size, image_size, 3))
    
    cnt = 0
    while cnt < num_of_pics_of_same_person:
        anchor_folder_index = 0
        neg_folder_index = 0
        while anchor_folder_index == neg_folder_index:
            anchor_folder_index =  random.randint(0, len(folder_list) - 1)
            neg_folder_index =  random.randint(0, len(folder_list) - 1)

        anchor_file_list = dict_[folder_list[anchor_folder_index]]
        anchor_index = 0
        pos_index = 0
        while anchor_index == pos_index:
            #print str(anchor_index)
            anchor_index = random.randint(0, len(anchor_file_list) - 1)
            pos_index = random.randint(0, len(anchor_file_list) - 1)

        neg_file_list = dict_[folder_list[neg_folder_index]]
        neg_index = random.randint(0, len(neg_file_list) - 1)
        
        anchor_file = anchor_file_list[anchor_index]
        pos_file = anchor_file_list[pos_index]
        neg_file = neg_file_list[neg_index]

        pos_pair_0 = (anchor_file, pos_file)
        pos_pair_1 = (pos_file, anchor_file)
        neg_pair_0 = (anchor_file, neg_file)
        neg_pair_1 = (neg_file, anchor_file)

        if pos_pair_0 not in set_ and pos_pair_1 not in set_ and neg_pair_0 not in set_ and neg_pair_1 not in set_:
            #print str(cnt)
            #print pos_pair_0
            set_.add(pos_pair_0)
            set_.add(neg_pair_0)
        else:
            continue

        hw_tuple = (image_size, image_size)
        file_list = [anchor_file, pos_file, neg_file]
        #print file_list
        img_list = []
        for file_ in file_list:
            temp = Image.open(file_)
            temp = temp.convert("RGB")
            temp = temp.resize(hw_tuple)
            img_list.append(temp)
        
        anchor_list[cnt, :, :, :] = preprocess_image(np.array(img_list[0]).astype(float))
        pos_list[cnt, :, :, :] = preprocess_image(np.array(img_list[1]).astype(float))
        neg_list[cnt, :, :, :] = preprocess_image(np.array(img_list[2]).astype(float))
        cnt += 1
        #print str(cnt)

    return anchor_list, pos_list, neg_list

def validate_per_epoch_with_threshold(model, image_size, anchor_list, pos_list, neg_list, threshold):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for i in range(len(anchor_list)):
        anchor_pic = anchor_list[i]
        pos_pic = pos_list[i]
        neg_pic = neg_list[i]

        anchor_arr = np.reshape(anchor_pic, (1, image_size, image_size, 3))
        pos_arr = np.reshape(pos_pic, (1, image_size, image_size, 3))
        neg_arr = np.reshape(neg_pic, (1, image_size, image_size, 3))
            
        anchor_predict = model.predict(anchor_arr, verbose = 0) 
        pos_predict    = model.predict(pos_arr, verbose = 0)
        neg_predict    = model.predict(neg_arr, verbose = 0)
            
        anchor_predict = anchor_predict[0][0][0]
        pos_predict    = pos_predict[0][0][0]
        neg_predict    = neg_predict[0][0][0]
            
        dist_pos = np.linalg.norm(anchor_predict - pos_predict)
        dist_neg = np.linalg.norm(anchor_predict - neg_predict)
            
        if dist_pos < threshold:
            true_pos += 1
        else:
            false_pos += 1
                
        if dist_neg < threshold:
            false_neg += 1
        else: true_neg += 1
    true_pos_rate = true_pos * 1.0 / ((true_pos + false_pos) * 1.0) 
    true_neg_rate = true_neg * 1.0 / ((true_neg + false_neg) * 1.0)
    return true_pos_rate, true_neg_rate
    

def validate_per_epoch(model, image_size, anchor_list, pos_list, neg_list):
    thred_low = 0.0
    thred_high = 100.0
    thred_mid = 0.0
    true_pos_rate = 0.0
    true_neg_rate = 0.0
    while thred_low < thred_high:
        thred_mid = thred_low + (thred_high - thred_low) / 2.0
        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0
        for i in range(len(anchor_list)):
            anchor_pic = anchor_list[i]
            pos_pic = pos_list[i]
            neg_pic = neg_list[i]

            anchor_arr = np.reshape(anchor_pic, (1, image_size, image_size, 3))
            pos_arr = np.reshape(pos_pic, (1, image_size, image_size, 3))
            neg_arr = np.reshape(neg_pic, (1, image_size, image_size, 3))
            anchor_predict = model.predict(anchor_arr, verbose = 0) 
            pos_predict    = model.predict(pos_arr, verbose = 0)
            neg_predict    = model.predict(neg_arr, verbose = 0)
            
            anchor_predict = anchor_predict[0][0][0]
            pos_predict    = pos_predict[0][0][0]
            neg_predict    = neg_predict[0][0][0]
            
            dist_pos = np.linalg.norm(anchor_predict - pos_predict)
            dist_neg = np.linalg.norm(anchor_predict - neg_predict)
            
            if dist_pos < thred_mid:
                true_pos += 1
            else:
                false_pos += 1
                
            if dist_neg < thred_mid:
                false_neg += 1
            else: true_neg += 1
        true_pos_rate = true_pos * 1.0 / ((true_pos + false_pos) * 1.0) 
        true_neg_rate = true_neg * 1.0 / ((true_neg + false_neg) * 1.0)
            
        #print "true_pos_rate: " + str(true_pos_rate)
        #print "true_neg_rate: " + str(true_neg_rate)
        #print "threshold: " + str(thred_mid)
            
        if true_pos_rate < true_neg_rate:
            thred_low = thred_mid + 0.0001
        else:
            thred_high = thred_mid
            
        if abs(true_pos_rate - true_neg_rate) < 0.05:
            break; 
                
    return true_pos_rate, true_neg_rate, thred_mid         

def filter_low_loss(anchor_list, pos_list, neg_list, anchor_label, pos_label, neg_label, model, discard, alpha, img_sz):
    picked_loss = []
    for i in range(len(anchor_list)):
        anchor = np.reshape(anchor_list[i], (1,img_sz,img_sz,3))
        pos = np.reshape(pos_list[i], (1,img_sz,img_sz,3))
        neg = np.reshape(neg_list[i], (1,img_sz,img_sz,3))
        X = {
                'anchor': anchor,
                'pos'   : pos,
                 'neg'   : neg
             }
        loss = model.predict(X)
        triloss = loss[0]
        posloss = loss[1]
        #print triloss.shape
        #print posloss.shape
        #print loss
        #print loss.shape
        if ((triloss[0] > 0 and triloss[0] < alpha) or (posloss[0] > 0 and triloss[0] < alpha)):
            picked_loss.append([triloss[0], posloss[0], i])
    #print picked_loss
    picked_loss.sort(key=operator.itemgetter(0))
    #print picked_loss
    index = picked_loss
    #index = sorted(picked_loss, key=p)
    #index = [item[1] for item in index[int(discard*len(index)):]]
    index = [item[2] for item in index]
    return anchor_list[index], pos_list[index], neg_list[index], anchor_label[index], pos_label[index], neg_label[index]
	
	
	
	
		
