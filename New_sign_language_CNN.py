import numpy as np
from keras.utils import to_categorical

fp_test = open("sign_mnist_test.csv", "r")
fp_train = open("sign_mnist_train.csv", "r")

twodDataTrain = []
twodDataTest = []


# reads the first line which is string pixel labels, information we dont want to use
trainFirstLine = fp_train.readline()

# goes through the training set and creates an array of 2d arrays
# the 2d arrays are an organization of the pixels into images
labelsTrain = []
for line in fp_train:
    longData = line.split(",")
    onePic = []
    label = np.zeros(25)
    label[int(longData[0])] = 1.0
    labelsTrain.append(label)
    for i in range(0, 28):
        oneRow = []
        firstIndex = (i * 28) + 1
        lastIndex = firstIndex + 28
        for j in range(firstIndex, lastIndex):
            oneRow.append(int(longData[j]))
        onePic.append(np.array(oneRow))
    twodDataTrain.append(np.array(onePic))

trainingData = np.array(twodDataTrain)

# reads the first line which is string pixel labels, this is information we dont want to use
testFirstLine = fp_test.readline()

# goes through the testing set and creates an array of 2d arrays
# the 2d arrays are an organization of the pixels into images
labelsTest = []
for line in fp_test:
    longDataTest = line.split(",")
    onePicTest = []
    labelTest = np.zeros(25)
    labelTest[int(longDataTest[0])] = 1.0
    labelsTest.append(labelTest)
    for i in range(0, 28):
        oneRowTest = []
        firstIndexTest = (i * 28) + 1
        lastIndexTest = firstIndexTest + 28
        for j in range(firstIndexTest, lastIndexTest):
            oneRowTest.append(int(longDataTest[j]))
        onePicTest.append(np.array(oneRowTest))
    twodDataTest.append(np.array(onePicTest))

labelsTrain = np.array(labelsTrain)
labelsTest = np.array(labelsTest)

trainingData = np.array(twodDataTrain)
testingData = np.array(twodDataTest)

print("this is training shape ", trainingData.shape)
print("this is testing shape", testingData.shape)
print("y train:", labelsTrain.shape)
print("y test:", labelsTest.shape)

x_train_vectors = (trainingData.reshape(27455, 28, 28, 1)) / 255.0
x_test_vectors = (testingData.reshape(7172, 28, 28, 1)) / 255.0

y_train_vectors = labelsTrain
y_test_vectors = labelsTest

print(" vector x train:", x_train_vectors.shape)
print(" vector x test:", x_test_vectors.shape)
print(" vector y train:", y_train_vectors.shape)
print(" vector y test:", y_test_vectors.shape)


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Construct the model
# The first layer must provide the input shape
neural_net = Sequential()
neural_net.add(Conv2D(16,(5,5),activation="relu",input_shape=(28,28,1)))
neural_net.add(MaxPooling2D(pool_size=(2,2)))
neural_net.add(Conv2D(32,(5,5),activation="relu",input_shape=(28,28,1)))
neural_net.add(MaxPooling2D(pool_size=(2,2)))
neural_net.add(Flatten())
neural_net.add(Dense(100, activation='relu'))
neural_net.add(Dense(25, activation='softmax'))
neural_net.summary()

# Compile the model
neural_net.compile(optimizer="SGD", loss="categorical_crossentropy",
                   metrics=['accuracy'])

# Train the model
history = neural_net.fit(x_train_vectors, y_train_vectors, verbose=1,
                         validation_data=(x_test_vectors, y_test_vectors),
                         epochs=7)
"""
running epoch above 20 seem to result in overfitting as val-loss starts to increases.
Overfitting becomes noticable at around 100 epochs. The value of val-loss varies
between independent runs, but epochs between 7-15 seem to be appropiate.
"""

loss, accuracy = neural_net.evaluate(x_test_vectors, y_test_vectors, verbose=0)
print("accuracy: {}%".format(accuracy*100))

# Examine which test data the network is failing to predict
import matplotlib.pyplot as plt
from numpy import argmax
from numpy.random import randint


# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.title('Model Loss')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

# Visualize Accuracy history
plt.plot(history.history['acc'], 'r--')
plt.plot(history.history['val_acc'], 'b-')
plt.title('Model Accuracy')
plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='upper left')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


# Examine which test data the network is failing to predict
outputs = neural_net.predict(x_test_vectors)
answers = [argmax(output) for output in outputs]
targets = [argmax(target) for target in y_test_vectors]
errorMap = {}



# The following code produces the Confusion Matrix
import pandas as pd
y_actu = pd.Series(targets, name='Actual')
y_pred = pd.Series(answers, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)


# setting up code for displaying what were the most recurring errors for each letter
for j in range(25):
    error = []
    errorMap[j] = error
for i in range(len(answers)):
    if answers[i] != targets[i]:
        errorMap[targets[i]].append(answers[i])

print("       ")
print("       ")
print("       ")
print("       ")

# maps errors to letters
confusionMatrix = {}
for i in range(25):
    numErrorsWThisLetter = []
    for j in range(25):
        numErrorsWThisLetter.append(0)
    confusionMatrix[i] = numErrorsWThisLetter
for i in range(25):
    if len(errorMap[i]) > 0: # there are errors for target i
        for error in errorMap[i]:
            confusionMatrix[i][error] += 1

# checks which 3 letters had the most total mistakes in the test run
from collections import Counter
high = Counter(confusionMatrix)
highest = high.most_common(3)
print("top three letters with most occuring errors:")
for p in highest:
    print(p[0], " : ", p[1], " ")


# displays each letters most recurring error
l= 0
k = 0
for i in range(25):
    max2 = 0
    max1 = 0
    for j in range(25):
        if max1 < confusionMatrix[i][j]:
            max2 = max1
            l = k
            max1 = confusionMatrix[i][j]
            k=j
        if max2 < max1 and max2 < confusionMatrix[i][j] and max1 != confusionMatrix[i][j]:
            max2 = confusionMatrix[i][j]
            l = j
    print("target", i, "'s most recuring error is on letter",k, ":", max1)
    print("target", i, "'s 2nd most recuring error is on letter",l, ":", max2)

# shows images for what the CNN incorrectly predicted
for i in range(len(answers)):
    if answers[i] != targets[i]:
        errorMap[targets[i]] = answers[i]
        print("Network predicted", answers[i], "Target is", targets[i])
        plt.imshow(testingData[i], cmap='gray')
        plt.show()
