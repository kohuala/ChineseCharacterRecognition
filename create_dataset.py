import os
import numpy as np
import struct
from PIL import Image
from data_extraction import read_from_gnt_dir


data_dir = ''
train_data_dir = os.path.join(data_dir, 'HWDB1.1trn_gnt')
test_data_dir = os.path.join(data_dir, 'HWDB1.1tst_gnt')

labels_character_set=np.load('character_dict.txt')
labels_character_list = np.load('character_list.pkl')
labels_character_set=set(labels_character_set)
labels_character_list=set(labels_character_list)

characters_to_recognize = np.load('characters_to_recognize.pkl')

char_dict = dict(zip(sorted(characters_to_recognize), range(len(characters_to_recognize))))

num_classes = 500
num_train = 60000
num_test = 10000

def create_dataset(X_input_training_data, y_output_training_data, X_input_testing_data, y_output_testing_data ):
    
    train_counter = 0
    test_counter = 0
    
    for image, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
        train_tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
        
        if train_tagcode_unicode in characters_to_recognize:
            
            if train_counter < num_train:
                im = Image.fromarray(image)
                im = im.resize((64,64))
                
                X_input_training_data.append(np.array(im.convert("L")))
                
                im.convert('L').save("train"+train_tagcode_unicode+str(train_counter) + '.png')
                y_output_training_data.append(char_dict[train_tagcode_unicode])
            train_counter+=1
    
    for image, tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):
        test_tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
        if test_tagcode_unicode in characters_to_recognize:
            if test_counter < num_test:
                
                im = Image.fromarray(image)
                im = im.resize((64,64))
                X_input_testing_data.append((np.array(im.convert("L"))))
                im.convert('L').save("test"+test_tagcode_unicode+str(test_counter) + '.png')
                                
                y_output_testing_data.append(char_dict[test_tagcode_unicode])
            test_counter+=1
  
    return X_input_training_data, y_output_training_data, X_input_testing_data, y_output_testing_data
      
def main():
    
    X_input_training_data, y_output_training_data, X_input_testing_data, y_output_testing_data = create_dataset([],[],[],[])
    np.save('X_input_training_data0', X_input_training_data)
    np.save('y_output_training_data0', y_output_training_data)
    np.save('X_input_testing_data0', X_input_testing_data)
    np.save('y_output_testing_data0', y_output_testing_data)
if __name__ =="__main__":
    main()           
    
