import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Bidirectional, Masking
from random import randint
import csv

"""Keras' Bidirectional layer wrapper takes an LTSM layer as an argument and
also the merge mode (how the forward and backward LTSM models are to be merged).
Merge options include; sum, mul, ave and concat. The default mode concat results
in double the number of outputs to the next layer, as the forward and backward
outputs are merged into a larger matrix"""

# file location, update for different users.
fileLocation = '/Users/nicolaschapman/Documents/UROP/Machine Leaning/Data/'

# read the data created by processByIonData.py into numpy arrays which are can be fed into the neural net.
with open(fileLocation + 'Network Data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    # initialise output list
    inputData = []
    outputData = []
    # initialise the list which will store the timestep data for a given peptide sequence.
    peptideInput = []
    peptideOutput = []
    for row in csv_reader:
        # skip the first peptide header in the csv
        if row[0] == 'Peptide 1':
            continue
        # when we reach a new peptide, append the peptideInput and peptideOutput to inputData and outputData
        if row[0][0:7] == "Peptide":
            timeSteps = len(peptideInput)
            # if the number of cleave sites (timesteps) is less than the max of 12 (as per a 15 amino peptide)
            # we need to pad peptideInput/Output with zero vectors for the remaining timesteps.
            if timeSteps != 12:
                for i in range(0, 12 - timeSteps):
                    inputPad = np.zeros(427)
                    outputPad = np.zeros(2)
                    peptideInput.append(inputPad)
                    peptideOutput.append(outputPad)
            # convert the peptideInput/Output to an array before it is added to input/outputData
            inputArray = np.array(peptideInput)
            inputData.append(inputArray)
            outputArray = np.array(peptideOutput)
            outputData.append(outputArray)
            peptideInput = []
            peptideOutput = []
            continue
        # rows in the csv have an input entry, then a corresponding output entry in the row after. We thus need
        # to differentiate b/w input and output entries and add them to the correct list.
        if len(row) == 2:
            # convert row to an array before it is appended to the relevant list.
            output = np.array([float(x) for x in row])
            peptideOutput.append(output)
        else:
            input = np.array([float(x) for x in row])
            peptideInput.append(input)
# convert input/outputData to a numpy array.
inputArray = np.array(inputData)
print(inputArray)
outputArray = np.array(outputData)
print(outputArray)
print(np.shape(inputArray))
print(np.shape(outputArray))

# split input array into train and test data
trainInput = inputArray[0:175]
trainOutput = outputArray[0:175]
testInput = inputArray[175:]
testOutput = outputArray[175:]

# define the LSTM model.
# define it as a sequential model to start
model = Sequential()

# Add masking layer
model.add(Masking(mask_value=0., input_shape=(12, 427)))

# add an initial hidden LSTM layer which has 60 memory units, and takes an input with 12
# time steps and 427 piece of information per time-step.
# Bidirectional wrapper simply tells it to go both ways, and prescribes a merge function.
# The memory units parameter is the dimensional size of the output per time step. It has very
# similar effects to the number of neurons in a feed-forward layer and has to be tuned to
# avoid overfitting while still ensuring enough information is stored to learn effectively.
model.add(Bidirectional(LSTM(60, return_sequences = True), merge_mode='concat'))

model.add(Bidirectional(LSTM(20, return_sequences = True), merge_mode='concat'))
# to just go in one direction:
# model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True))
# to go in the reverse direction:
# model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True, go_backwards=backwards))

# Output layer is a fully connected dense layer that outputs one output per time step.
# It uses the sigmoid activation function.
# The TimeDistributes wrapper appears to be arbitrary but I have left it in.
model.add(Dense(2, activation = 'sigmoid'))
#model.add(Bidirectional(LSTM(1, return_sequences = True), merge_mode='sum'))

# compile the model with binary crossentropy cost function (because the output is either
# 1 or 0) the standard keras optimiser (backprop algorithm) of adam and return accuracy
# after each epoch.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['acc'])

# Produce 1000 sequences and train over 20 epochs with a mini-batch size of 50
# verbose param sets how information regarding epoch progression is presented in
# the console.
model.fit(trainInput,trainOutput, epochs=20, batch_size=10, verbose=2)

# evaluate LTSM
testData = zip(testInput, testOutput)

for input, output in testData:
    input = np.reshape(input, (1,12,427))
    outputPredicted = model.predict(input, verbose=1)
    print('Predicted Output: ')
    print(outputPredicted)
    print('Expected Output: ')
    print(output)