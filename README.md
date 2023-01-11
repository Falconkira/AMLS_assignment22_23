# AMLS_assignment22_23
This is an implementation of AMLS 22-23 final project.

A1, A2, B1 and B2 is the folder for the python implementation of each task. The excel file in the folders is the cross-validation results and graph for each model. The jupyter notebook in each file serves as a draft for the .py file. It is only used for testing.

The Datasets folder is empty due to the submission requests. It should contain the following datasets to run the code.
• cartoon_set
• celeba
• cartoon_set_test 
• celeba_test

The results.xlsx store all the results of models, which can also be found in reports with explanations.

The dat file is for face features extraction.

Simply running main.py to check the results of all applications. In order to run the file, it requires several python libraries:
• dlib
• tensorflow
• os
• sklearn
• pandas
• numpy
• cv2
The code is originally runs in windows. If want to run it in Mac, the file_name in extract_features_labels function (in A1.py, A2.py, B1.py and B2.py) need to be changed. There is a difference on the deafult path represenatation on different systems, where windows use \\, and Mac use /. There is also a comment in each file to show the correct implementation for each system.
