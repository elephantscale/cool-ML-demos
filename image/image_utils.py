## Utility functions 

# %matplotlib inline



def sample_from_dict(d, sample=10):
    import random
    
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))
## -- end: sample_from_dict()

def get_index_for_value_from_dict(d, value):
    for i, (k,v) in enumerate(d.items()):
        if v == value:
            return k
    return None

def get_image_files_from_dir(directory, recursive=True):
    import os, glob
    
    # grab image files:   *.jpeg,   *.jpg,   *.png  (ending in 'g')
    listing = glob.glob(os.path.join(directory, "**/*.*g"), recursive=recursive)
    files = [f for f in listing if os.path.isfile(f)]
    return files

## Gets number of files in the directory and total size in MB
def get_dir_stats(adir):
    import os
    
#     listing = glob.glob(os.path.join(adir, "**/*.*"), recursive=True)
#     files = [f for f in listing if os.path.isfile(f)]
    image_files = get_image_files_from_dir(adir, recursive=True)
    file_sizes = [os.path.getsize(f) for f in image_files]
    total_file_size_MB = round(sum(file_sizes) / (1024*1024), 2)
    return (len(image_files), total_file_size_MB)
    

def print_dir_stats (a_dir_name, adir, a_class_labels):
    import os 
    
    dir_stats = get_dir_stats(adir)
    print ('--- {} ({}):  files={},  size={} MB'.format(a_dir_name, adir, dir_stats[0], dir_stats[1]))
    for class_label in a_class_labels:
        class_dir = os.path.join(adir, class_label)
        dir_stats = get_dir_stats (class_dir)
        print ('       +-- {} :  files={},  size={} MB'.format(class_label, dir_stats[0], dir_stats[1]))


def get_class_labels(a_training_dir):
    import os 
    return [d for d in os.listdir(a_training_dir) if os.path.isdir(os.path.join(a_training_dir,d))]
        
def print_training_validation_stats (a_training_dir, a_validation_dir):
    class_labels = get_class_labels(a_training_dir)
    print ('Found class lables:', class_labels)
    print ()

    print_dir_stats('training_data', a_training_dir, class_labels)
    print()
    if a_validation_dir:
        print_dir_stats('validation_data', a_validation_dir, class_labels)
        
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plot_images(images_arr, message=""):
    import matplotlib.pyplot as plt
    
    if message:
        plt.suptitle( message, fontsize=20, fontweight='bold')
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
#     plt.tight_layout()
    plt.show()
    

def display_images_from_dir (image_dir, num_images_per_label=5):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import random
    import os
    
    class_labels = get_class_labels(image_dir)
    
    fig_rows = len(class_labels)
    fig_cols = num_images_per_label + 1  # adding 1 to columns, for text labels
    
    fig = plt.gcf()
    fig.set_size_inches(fig_cols * 3, fig_rows * 3)
    #     fig.subplots_adjust(hspace=0.4, wspace=0.4)

    plt.suptitle( "images from : " + image_dir, fontsize=20, fontweight='bold')
    plt.axis('off')
    
    row = 0
    index = 0
    for label in class_labels:

        class_dir = os.path.join(image_dir, label)
        class_file_listing = os.listdir(class_dir)

        random_class_images = random.sample(class_file_listing, num_images_per_label )
       
        row = row + 1
        index = index + 1
        sp = plt.subplot(fig_rows, fig_cols, index)
        sp.text (0.5,0.5, label, fontsize=18, ha='center')
        sp.axis('Off') # Don't show axes (or gridlines)
   
        for img_file in random_class_images:
            index = index + 1
            sp = plt.subplot(fig_rows, fig_cols, index)
            sp.axis('Off') # Don't show axes (or gridlines)
            
            img_file_path = os.path.join(class_dir, img_file)
            img = mpimg.imread(img_file_path)
            plt.imshow(img)
            
            # this will print image file name
            # sp.text(0,0, img_file)
        
        
    plt.show()
# ---------- end : display_images_from_dir



## Calcutes a table for each image with these stats:
##   'softmax_output'
##   'max_probability'
##   'predicted_class'
##   'expected_class'
## Used by many other plotting functions
def calculate_prediction_table(model, data_gen):
    from math import ceil
    import numpy as np
    
    ground_truth = test_labels = data_gen.classes
    data_gen.reset()  # revert back to batch 1
    predictions = model.predict(data_gen, batch_size=data_gen.batch_size, 
                                      steps=ceil(data_gen.n / data_gen.batch_size) )
    ## Ensure all predictions match
    assert(len(predictions) == len(ground_truth) )
    
    #prediction_table = {}  # dict
    prediction_table = []  # array
    for index, softmax_array in enumerate(predictions):
        index_of_highest_probability = np.argmax(softmax_array)
        value_of_highest_probability = softmax_array[index_of_highest_probability]
        prediction_entry = { 'softmax_output' : softmax_array,
                                    'max_probability' : value_of_highest_probability, 
                                    'predicted_class' : index_of_highest_probability, 
                                    'expected_class' : ground_truth[index], 
                                    'img_name' : data_gen.filenames[index],
                                    'img_file' : data_gen.filepaths[index]
                                  }
        # prediction_table[index] = prediction_entry # dict
        prediction_table.append(prediction_entry) # array
    
    return prediction_table
#  ------   end : calculate_prediction_table






## ----------------------------
# Helper function that finds images that are closest
# Input parameters:
#   prediction_table: dictionary from the image index to the prediction
#                      and ground truth for that image
#   get_highest_probability: boolean flag to indicate if the results
#                            need to be highest (True) or lowest (False) probabilities
#   label: id of category
#   number_of_items: num of results to return
#   only_false_predictions: boolean flag to indicate if results
#                           should only contain incorrect predictions
def get_images_with_sorted_probabilities(prediction_table, get_highest_probability,
                                         label, number_of_items, only_false_predictions=False):
    import pprint
    
    # dict
#     sorted_prediction_table = sorted(prediction_table.items(), 
#                                      key=lambda x: x[1].get('max_probability'), 
#                                      reverse = get_highest_probability)
    
    # array
    sorted_prediction_table = sorted(prediction_table, 
                                     key=lambda x: x['max_probability'], 
                                     reverse = get_highest_probability)
    
    # pprint.pprint (sorted_prediction_table)
    result = []
    #for index, (k,v) in enumerate(sorted_prediction_table):  # dict
    for index, v  in enumerate(sorted_prediction_table):
        # image_index = k
        max_probability = v.get('max_probability')
        predicted_index = v.get('predicted_class')
        expected_index = v.get('expected_class')
        
        if predicted_index == label:
            if only_false_predictions == True:
                if predicted_index != expected_index:
                    #result.append([image_index, v ])
                    result.append(v)
            else:
                #result.append([image_index, v ])
                result.append(v)
        if len(result) >= number_of_items:
            break
    # end for        
    return result
## --- end: get_images_with_sorted_probabilities()





## ----------------------
def plot_image_predictions (prediction_table, message, label_mappings={}):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import os
    from math import ceil
    
    columns = 5
    rows = ceil(len(prediction_table) / columns)

    
    fig = plt.gcf()
    plt.figure(figsize=(columns*5, rows*6))
#     fig.set_size_inches(columns * 4, 1.5 + rows * 4)
#     fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    plt.suptitle( message, fontsize=20, fontweight='bold')
    plt.axis('off')

    # for i, (index, result) in enumerate(prediction_table.items()): # if dict
    for index, result in enumerate(prediction_table):  # if array
        # print ('i:', i, ', index:', index, ',  result:', result)
        predicted_class = result['predicted_class']
        probability = result['max_probability']
        predicted_label = get_index_for_value_from_dict(label_mappings, predicted_class)
        predicted_label_str = "("+predicted_label+")" if predicted_label else ""
        #image_name = os.path.basename(result['img_file'])
        image_name = result['img_name']
        image_file = result['img_file']
        img = mpimg.imread(image_file)

        
        ax = plt.subplot(len(prediction_table) / columns + 1, columns, index + 1)
        ax.axis('off')       
        ax.set_title("\n\n{}\nPredicted: {} {}\nProbability:{:.2f}".format(image_name, predicted_class, predicted_label_str, probability))
        plt.imshow(img)
    
## --- end: plot_image_predictions




def predict_on_images_in_dir (model, image_dir, image_width, image_height):
    import random
    import numpy as np
    from tensorflow.keras.preprocessing import image
    import pprint
    from math import ceil

    image_files = get_image_files_from_dir(image_dir)
    
    prediction_table = predict_on_images(model, image_files, image_width, image_height)
    return prediction_table
## ---- end : predict_on_images_in_dir()


## Predicts on given image files... 
## does not use a generator
def predict_on_images(model, files, image_width, image_height):
    import os
    import random
    import numpy as np
    from tensorflow.keras.preprocessing import image
    import pprint

    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    # prediction_results = {} # dict
    prediction_results = [] # array
    
    for index, file in enumerate(files):
        img = image.load_img(file, target_size = (image_width, image_height))
        # print (image_file)
        img_data = image.img_to_array(img) / 255.
        # option 1 reshape
        #img_data = np.expand_dims(img_data, axis = 0)
        #prediction = model.predict (image_data)
        # or option 2: no reshape and predict
        prediction = model.predict(img_data[None]) # this is softmax
        
        index_of_highest_probability = np.argmax(prediction[0])
        value_of_highest_probability = prediction[0][index_of_highest_probability]
        # print (prediction)

        x = {
            'img_file': file,
            'img_name' : os.path.basename(file),
            'softmax_output' : prediction, 
            'max_probability' : value_of_highest_probability, 
            'predicted_class' : index_of_highest_probability, 
            }
        # pprint.pprint (x)
        #prediction_results[index] = x  # dict
        prediction_results.append(x)
    # end for
    return prediction_results
## ---- end : predict_on_random_images()


## This will go through all classes, and predict high/low probabilities
def plot_prediction_stats_on_all_classes (model, data_gen):
    prediction_table = calculate_prediction_table(model, data_gen)
    # pprint (sample_from_dict (prediction_table, 5))

    data_gen.reset()
    print ("label indices : ", data_gen.class_indices)
    
    image_dir = data_gen.directory
    print ("image directory : ", image_dir)
    
    for label_index, label  in enumerate(data_gen.class_indices):
        print(label, label_index)

        ## Find the label classification with highest confidence
        message = "Images of '{}' (label_index={}) with the highest confidence ".format(label, label_index)
        highest_confidence_images = get_images_with_sorted_probabilities(prediction_table=prediction_table, 
                                                                         get_highest_probability=True, 
                                                                         label=label_index, number_of_items=10, 
                                                                         only_false_predictions=False)
#         display_image_predictions(image_dir=image_dir, sorted_results=highest_confidence_images[:10], 
#                                   predicted_index=label,  message=message, fnames=data_gen.filenames,
#                                   label_mappings=data_gen.class_indices)
        
        plot_image_predictions(prediction_table=highest_confidence_images, message=message, label_mappings=data_gen.class_indices )


        ## Find the label classifications with lowest confidence
        message = "Images of '{}' (label_index={}) with the lowest confidence ".format(label, label_index)
        lowest_confidence_images = get_images_with_sorted_probabilities(prediction_table=prediction_table, 
                                                                        get_highest_probability=False, 
                                                                        label=label_index, number_of_items=10, 
                                                                        only_false_predictions=False)
#         display_image_predictions(image_dir=image_dir, sorted_results=lowest_confidence_images[:10], 
#                                   predicted_index=label, message=message, fnames=data_gen.filenames,
#                                   label_mappings=data_gen.class_indices)
        plot_image_predictions(prediction_table=lowest_confidence_images, message=message, label_mappings=data_gen.class_indices)



        ## mis-predicted
        message = "Images that are mis-predicted as '{}' (label_index={})".format(label, label_index)
        mis_predicted_images = get_images_with_sorted_probabilities(prediction_table=prediction_table, 
                                                                    get_highest_probability=True, 
                                                                    label=label_index, number_of_items=10,
                                                                    only_false_predictions=True)
#         display_image_predictions(image_dir=image_dir, sorted_results=mis_predicted_images[:10], 
#                                   predicted_index=label, message=message, fnames=data_gen.filenames,
#                                   label_mappings=data_gen.class_indices)
        plot_image_predictions(prediction_table=mis_predicted_images, message=message, label_mappings=data_gen.class_indices)

## ----- end : plot_prediction_on_all_classes