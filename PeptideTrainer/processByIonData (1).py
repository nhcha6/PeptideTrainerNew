from pyteomics import mgf
import csv
from bisect import bisect_left
import sklearn.preprocessing as preprocessing
import numpy as np
import math
import table
import csv

H20_MASS = 18.010565

# Mono-isotopic mass
monoAminoMass = {
    'A': 71.03711,
    'R': 156.10111,
    'N': 114.0493,
    'D': 115.02694,
    'C': 103.00919,
    'E': 129.04259,
    'Q': 128.05858,
    'G': 57.02146,
    'H': 137.05891,
    'I': 113.08406,
    'L': 113.08406,
    'K': 128.09496,
    'M': 131.04049,
    'F': 147.06841,
    'P': 97.05276,
    'S': 87.03203,
    'T': 101.04768,
    'W': 186.07931,
    'Y': 163.06333,
    'V': 99.06841,

}

aminoIndex = {
    'A': 0,
    'R': 1,
    'N': 2,
    'D': 3,
    'C': 4,
    'E': 5,
    'Q': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'L': 10,
    'K': 11,
    'M': 12,
    'F': 13,
    'P': 14,
    'S': 15,
    'T': 16,
    'W': 17,
    'Y': 18,
    'V': 19,
}

fileLocation = '/Users/nicolaschapman/Documents/UROP/Machine Leaning/Data/'
csvPath = [fileLocation + 'VACVpeps-rackA_peptides.csv',
           fileLocation + 'VACVpeps-rackB_peptides.csv']
mgfPath = [fileLocation + 'VACVpeps-rackA.mgf',
            fileLocation + 'VACVpeps-rackB.mgf']

def readData(mgfList, csvList):
    scanNumbers = {}
    byIonDict = {}

    # zip mgfPath and csvPath together so that they can be iterated across simultaneously
    iterable = zip(mgfList, csvList)
    # loop through each pair of mgf and csv and extract the required data
    for mgfPath, csvPath in iterable:
        # create dictionary with scan number as key and peptide as value from the csv file
        with open(csvPath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[9] == 'Scan':
                    continue
                # ignore if there are numbers (implying modification) or if the length is greater than 15
                if any(char.isdigit() for char in row[0]) or len(row[0]) > 15:
                    continue
                # ignore if the first b/y ion is the same as the last, as this is causing issues for now.
                # will definitely look to incorporate this occurance in the future.
                if row[0][0:2] == row[0][-2:]:
                    continue
                scanNumbers[row[9]] = row[0]

        # create a dictionary with the peptide as the key and [[m/z array],[intensity array]] as the value.
        with mgf.read(mgfPath) as mgfReader:
            for spectrum in mgfReader:
                scan = spectrum['params']['scans']
                if scan in scanNumbers.keys():
                    byIonDict[scanNumbers[scan]] = [spectrum['m/z array'], spectrum['intensity array'], spectrum['params']['pepmass'][0]]
                    #break
    return byIonDict

# calculate the mass of the b/y ions for each peptide filter the m/z array and intensity array so that it only
# contains b/y ions
def findByIonIntensities(byIonDict):
    peptideIntensityDict = {}
    for peptide, arrays in byIonDict.items():
        # calculate the sum of the intensities for later normalisation calculation
        intensitySum = 0
        intensityArray = arrays[1]
        for intensity in intensityArray:
            intensitySum += intensity
        # obtain precursor mass:
        precursorMass = arrays[2]
        # calculate byIons
        blist, ylist = createBYIons(peptide)
        byDict = ionMassDict(blist, ylist)
        mzArray = arrays[0]
        # normalise the intensity array data to add to 1 using l1 normalisation.
        intensityArray = np.reshape(intensityArray, (1,-1))
        normIntensArray = preprocessing.normalize(intensityArray, norm='l1')
        # initialise the first element of intensityTups to be the precursor mass.
        ionIntensityTups = [precursorMass]
        for ion, mz in byDict.items():
            # Note that a handful of ions do not have a good match on mass, possibly needs reviewing. Currently
            # we are ignoring ions with only 1 aminoacid as they showed the most erratic results.
            closestIndex = takeClosest(mzArray, mz, True)
            #closestMz = mzArray[closestIndex]
            normalisedIntensity = normIntensArray[0][closestIndex]
            ionIntensityTups.append((ion, normalisedIntensity, mz))
        peptideIntensityDict[peptide] = ionIntensityTups
    return peptideIntensityDict

def normaliseLog(rawIntensity, intensitySum):
    temp = rawIntensity/intensitySum
    normalised = math.log(temp)
    return normalised

def bMassCalc(peptide, modlist = None):
    mass = 1
    for char in peptide:
        if char.isalpha():
            char = char.upper()
            mass += monoAminoMass[char]
        else:
            mod = modlist[int(char)-1]
            mass += modTable[mod][-1]
    return mass

def yMassCalc(peptide, modlist = None):
    mass = H20_MASS + 1
    for char in peptide:
        if char.isalpha():
            char = char.upper()
            mass += monoAminoMass[char]
        else:
            mod = modlist[int(char)-1]
            mass += modTable[mod][-1]
    return mass

def ionMassDict(blist,ylist):
    dict = {}
    for i in range(0,len(blist)):
        pepB = blist[i]
        pepY = ylist[i]
        dict[pepB] = bMassCalc(pepB)
        dict[pepY] = yMassCalc(pepY)
    return dict

def createBYIons(peptide):
    blist = []
    ylist = []
    #changed this for loop slightly to ignore ions which contain only one amino acid, as these
    for i in range (1,len(peptide)-2):
        b = peptide[0:i+1]
        y = peptide[i+1:]
        blist.append(b)
        ylist.append(y)
    return blist, ylist

def takeClosest(myList, myNumber, indexBool = False):
    """
    Assumes myList is sorted. Returns closest value to myNumber via index.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        if indexBool:
            return 0

        return myList[0]
    if pos == len(myList):

        if indexBool:
            return -1

        return myList[-1]

    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        if indexBool:
            return pos
        return after

    else:
        if indexBool:
            return pos - 1
        return before

def pepToVector(peptide):
    array = np.array([])
    len = 0
    for char in peptide:
        len += 1
        toAdd = np.zeros((20))
        index = aminoIndex[char]
        toAdd[index] = 1
        array = np.append(array, toAdd)
    # fill out array to max length of 15
    if len != 15:
        rem = 15 - len
        addElements = rem*20
        array = np.append(array, np.zeros((addElements)))
    # divide by max allowable to get peptideLen between 0 and 1
    normLen = len/15
    return array, normLen

#retruns an output of 1 if the intensity is greater than 1% of the sum of the intensities, 0 if not.
def thresholdOutput(bInt, yInt):
    if bInt > 0.005:
        bOutput = 1
    else:
        bOutput = 0
    if yInt > 0.005:
        yOutput = 1
    else:
        yOutput = 0
    return [bOutput, yOutput]

def createPepInputVector(peptide, tups):
    # calculate and normalise precursor mass. Normalised with respect to the max mass possible for a peptide of
    # that length. May also look to simply input the number as opposed to normalising it.
    normPrecMass = tups[0]/(186.07931*len(peptide))
    #print('precursor mass is: ')
    #print(tups[0])

    # convert precursor sequence to vector and length
    # May also look to simply input the precursorLen as a the real number as opposed to normalising it.
    precursorVector, normPrecursorLen = pepToVector(peptide)
    #print('precursor length is: ')
    #print(len(peptide))
    #print('precursor sequence is: ')
    #print(peptide)

    # convert first and last aminoAcids to vectors
    firstAminoVector = np.zeros(20)
    firstAminoVector[aminoIndex[peptide[0]]] = 1
    lastAminoVector = np.zeros(20)
    lastAminoVector[aminoIndex[peptide[-1]]] = 1
    #print('first amino: ')
    #print(peptide[0])
    #print('last amino: ')
    #print(peptide[-1])

    # all cleavage site inputs from a single peptide will contain the following data:
    pepInputVector = np.append(normPrecursorLen, normPrecMass)
    pepInputVector = np.append(pepInputVector, precursorVector)
    pepInputVector = np.append(pepInputVector, firstAminoVector)
    pepInputVector = np.append(pepInputVector, lastAminoVector)
    #print(pepInputVector)
    return pepInputVector

def createCleaveVector(pepInputVector, peptide, bIonData, yIonData, cleaveNumber):
    # define the cleave intensity output
    cleaveOutput = thresholdOutput(bIonData[1], yIonData[1])
    #print('b/y ion intensities: ')
    #print(bIonData[1])
    #print(yIonData[1])

    # the input for each cleavage site begins with the same vector of information on the precursor ion.
    cleaveInput = pepInputVector

    # extract the first set of b/y ions from the list of tups.
    bIonSeq = bIonData[0]
    yIonSeq = yIonData[0]
    #print('b and y ion sequences: ')
    #print(bIonSeq)
    #print(yIonSeq)

    # calculate and normalise the byIonLength. byIonLength normalised with respect to the max precursor
    # length of 15, which has a max b/y ion length of 13 and min of 2.  We may look to alter this normalisation
    # to make it with reference to the specific percursor length (as was done with the cleaveNumber)
    # as opposed to the max precursor length.
    # May also look to simply input the number as opposed to normalising it.
    bLen = len(bIonSeq)
    normbLen = bLen / 13
    yLen = len(yIonSeq)
    normyLen = yLen / 13
    ionLens = [normbLen, normyLen]
    #print('b and y ion lengths: ')
    #print(bLen)
    #print(yLen)

    # calculate and normalise the b/y ion masses. Normalised with respect to the maximum mass of an ion
    # which could be derived from a 15 length peptide. This is equal to 13 of the amino acids with highest mass
    # as a sequence. May look to simply use the real mass values instead.
    bMass = bIonData[2]
    bMassNorm = bMass / (13 * 186.07931)
    yMass = yIonData[2]
    yMassNorm = yMass / (13 * 186.07931)
    ionMasses = [bMassNorm, yMassNorm]
    #print('b and y ion masses: ')
    #print(bMass)
    #print(yMass)

    # normalise cleave number, add it to cleaveSiteVect, and add 1 to cleaveNumber for next iteration.
    # currently being normalised with reference to the length precursor, where cleavages can
    # be made from the 2nd to the len - 2 cleaveNumber. May look to normalise with regard to the length of
    # the max precursor as is done with ion len, however I have chosen not to as the this piece of data seemed
    # to represent how far through the peptide the cleave has occured.
    # May also look to simply input the number as opposed to normalising it.
    normCleaveNum = cleaveNumber / len(peptide)
    #print('cleave number: ')
    #print(cleaveNumber)

    # initialise new variable for the cleave site neighbours.
    cleaveSiteNeighbours = []
    for i in range(1, 3):
        cleaveSiteNeighbours.append(bIonSeq[-i])
        cleaveSiteNeighbours.append(yIonSeq[i - 1])
    #print('cleave site first neighbours: ')
    # print(cleaveSiteNeighbours[0])
    # print(cleaveSiteNeighbours[1])
    # print('cleave site second neighbours: ')
    # print(cleaveSiteNeighbours[2])
    # print(cleaveSiteNeighbours[3])

    # print(cleaveSiteNeighbours)
    # initialise array to hold data specific to the cleaveSite, and add ionLens, ionMasses and cleave number to it.
    cleaveSiteVect = np.array([])
    cleaveSiteVect = np.append(ionLens, ionMasses)
    cleaveSiteVect = np.append(cleaveSiteVect, normCleaveNum)
    # convert cleave site neighbours from amino acids to vectors
    for amino in cleaveSiteNeighbours:
        zeros = np.zeros(20)
        zeros[aminoIndex[amino]] = 1
        cleaveSiteVect = np.append(cleaveSiteVect, zeros)

    # append cleaveSiteVectors to cleaveInput
    cleaveInput = np.append(cleaveInput, cleaveSiteVect)
    return cleaveInput, cleaveOutput

def createData(peptideIntensityDict):
    # initialise lists to store input and output data
    inputData = []
    outputData = []
    # loops through each peptide to build input and output data
    for peptide, tups in peptideIntensityDict.items():
        # initialise peptide arrays
        peptideInput = []
        peptideOutput = []
        # call function to return the vector which compiles the data relevant to all cleave sites within
        # a given peptide.
        pepInputVector = createPepInputVector(peptide, tups)
        # counter to store location of cleave site
        cleaveNumber = 2
        # now loop through to build data relevant only to individual cleavage sites
        for i in range(1, len(tups), 2):
            #print(tups[i])
            #print(tups[i+1])
            # call function which adds to pepInputVector to create the entire input for a given cleave site.
            # also returns the desired output for that cleave.
            cleaveInput, cleaveOutput = createCleaveVector(pepInputVector, peptide, tups[i], tups[i+1], cleaveNumber)
            # add 1 to cleaveNumber for next iteration
            cleaveNumber += 1
            # append the input for the cleaveSite to the input data, and cleaveOutput to output
            peptideInput.append(cleaveInput)
            peptideOutput.append(cleaveOutput)
        inputData.append(peptideInput)
        outputData.append(peptideOutput)
    return inputData, outputData

def writeToFile(inputData, outputData):
    iterator = zip(inputData, outputData)
    for input, output in iterator:
        with open( + '.csv', 'a', newline='') as csv_file:

            writer = csv.writer(csv_file, delimiter=',')
            if groupedBy is 'ProtToPep':
                header = 'Protein'
            else:
                header = 'Peptide'
            writer.writerow([header])
            for key, value in seenPeptides.items():
                infoRow = [key]
                writer.writerow(infoRow)
                for peptide in value:
                    writer.writerow([peptide])
                writer.writerow([])

def writeDataToCsv(input, output):
    with open(fileLocation + "Network Data.csv", mode = 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        peptideIter = zip(input, output)
        pepNumber = 1
        for inputList, outputList in peptideIter:
            headerRow = "Peptide " + str(pepNumber)
            pepNumber += 1
            writer.writerow([headerRow])
            cleaveIter = zip(inputList, outputList)
            for cleaveInput, cleaveOutput in cleaveIter:
                inputRow = []
                for val in cleaveInput:
                    inputRow.append(val)
                writer.writerow(inputRow)
                outputRow = cleaveOutput
                writer.writerow(outputRow)


    # with open(outputPath+ groupedBy + '.csv', 'a', newline='') as csv_file:
    #
    #     writer = csv.writer(csv_file, delimiter=',')
    #     if groupedBy is 'ProtToPep':
    #         header = 'Protein'
    #     else:
    #         header = 'Peptide'
    #     writer.writerow([header])
    #     for key, value in seenPeptides.items():
    #         infoRow = [key]
    #         writer.writerow(infoRow)
    #         for peptide in value:
    #             writer.writerow([peptide])
    #         writer.writerow([])

# read the mgf and csv data to create a scanNumbers Dict and a byionDict
dataDict = readData(mgfPath, csvPath)
# print('Data after initial read = ')
# print(dataDict)

# return a dict which has peptides as keys, and tuples containing b/y ion sequence, mz and intensity data
peptideIntensityDict = findByIonIntensities(dataDict)
#print("peptideIntensityDict = ")
#print(peptideIntensityDict)

inputData, outputData = createData(peptideIntensityDict)
print("Input Data = ")
print(inputData)
print(np.size(inputData))
print("Output Data = ")
print(outputData)
print(np.size(outputData))

writeDataToCsv(inputData, outputData)



# #normalisation code from a module dedicated to preprocessing for machine learning
# list = []
# for i in range(1,21):
#     list.append(i)
# print(list)
# list = np.reshape(list, (1,-1))
# print(list
#
# normList = preprocessing.normalize(list, norm='max')
#
# print(normList)
