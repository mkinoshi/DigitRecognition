# Digit regonition system using neural network 

This is my first code of neuran network in Matlab. I used the data from Kaggle, which is pretty popular and easy to use for a biginner. I built a simple neural network to detect a digit from images. This code does not include the code of separting the trainin dataset and the test dataset. The training dataset includes data points, and the test dataset includes data points. This code is intented to show how a simple neural network works.  

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Install Matlab (obviously) 
Obtain datasets from the following link: (https://www.kaggle.com/c/digit-recognizer/data) 
```


## Running the tests
This matlab code contains one main function and multiple function files.
```
Run proj_main.m
```

### Break down into end to end tests

First, this code loads and displays a sample of images. 

Second, this code initializes the weights randomly, and start training the dataset. This neural network using a sigmoid function as a activation function. Hidden layer size is one of tha parameters that we have to adjust, and you can change its size by changing the value of hidden_layer_size parameter. Since the goal of this neural network is to recoginize the hand written digit, the output of this neural network is from 0 to 9. This program also visualizes the result of training after a certain number of iterations. Below, there are two images; one is after 10 iterations and the other is after 100 iterations. As you can see, the neural network learns features from images as it iterates more. It takes aroudn 10 mintues to finish iterating 100 times.  

Third, it runs on the test dataset using the trained neural network. It will create a csv file and save it as 

On this project, I implemented all of the functions including sigmoid funcions and sigmoid gradient functions for the purpose of learning the mechanism of neural network. Regardless of the intimidating name, a neural network system is pretty simple. To 


## Acknowledgments

* Kaggle community for providing the data
