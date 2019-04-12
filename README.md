# Handwritten Chinese Character Recognition

** Abstract **
This project focuses on deep neural network technique to recognize isolated offline handwritten Chinese characters from the Institute of Automation of Chinese Academy of Sciences (CASIA) database. Five neural network architectures were explored including Dr. Yann LeCun’s LeNet-5 network designed for handwritten and machine-printed character recognition and the multi-column deep neural network (DNN). The experimental results are recorded for a 500-class dataset with my network model. The role of normalization for better accuracy will also be discussed.


** Introduction **
Deep convolutional neural network (CNN) has already become the technique for recognition problems in the recent years. For example, one can find literature, research, and results on Modified National Institute of Standards and Technology database (MNIST) for handwritten digits. Although recognizing digits and Chinese characters are indeed classification problems, Chinese character classification presents to be much, much more difficult. Many reasons include many more categories: the digits classification has only 10 classes whereas there are over 50,000 Chinese characters but only 3500 used for day to day purposes. Also, there is exists more variance in the characters. This can be seen in the below figure. As seen, there are many ways to write the character 风  by different writers and even between the same writer. Moreover, the character can be mistaken as another character such as 凡 even by a human reader when the character is isolated. The current record for handwritten recognition of 3755 classes is 96.7% produced by Fujitsu in 2006.
