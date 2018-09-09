from nltk.corpus import wordnet
from nltk.corpus import stopwords
import random

# Parses a sentence and returns that processed sentence, with a list of its
# synsets
def parseTextToValuableSynsetsAsList(text):
    #tokenizer = RegexpTokenizer(r'\w+')
    tokenizedText = text #tokenizer.tokenize(text)
    stopWords = set(stopwords.words('english'))
    stopWords.add("I")
    wordsWithSynsets = list()
    #This holds a list that either has the corresponding set of synsets for the
    # word in the input text or None if none found
    sentence = list()
    sentenceSynsets = []
    currentSentence = text
    for word in currentSentence:
        wordSynsets = wordnet.synsets(word)
        if wordSynsets != []:
            wordSynsets.insert(0, word)
            sentenceSynsets.append(wordSynsets)
            sentence.append(True)
        else:
            sentence.append(False)
    return (sentence,sentenceSynsets)

# Different ways of measuring similarity between synsets
similarityMeasures = [wordnet.wup_similarity, wordnet.path_similarity]

# Compares two synsets and stores comparison data
def compareSynset(synsetToCompare, synsetList, simMeasureIndex):
    avgScore = 0
    maxScore = 0
    maxSynset = 0
    simToTopSense = 0
    for synset in synsetList:
        similarity = similarityMeasures[simMeasureIndex](synsetToCompare,synset)
        if similarity != None:
            avgScore += similarity
            if similarity > maxScore:
                maxScore = similarity
                maxSynset = synset
    avgScore /= len(synsetList)
    topSenseSim = similarityMeasures[simMeasureIndex](synsetToCompare,
                                                                  synsetList[0])
    if topSenseSim != None:
        simToTopSense = topSenseSim
    return [avgScore, (maxScore, maxSynset),simToTopSense]

# Compares multiple synsets bounded by window size in the original sentence and
# returns that comparison data
def compareAllSynsets(windowSize, simMeasureIndex, sentenceSynsetData):
    synsetComparisonData = {}

    for row in sentenceSynsetData:
        rowLength = len(row)
        for inRowIndex in range(1, rowLength):
            currentRowIndex = sentenceSynsetData.index(row)
            leftLimit = currentRowIndex - windowSize
            rightLimit = currentRowIndex + windowSize
            for rowIndex in range(leftLimit,rightLimit+1):
                if (rowIndex != currentRowIndex) & (rowIndex >= 0) & \
                                        (rowIndex < len(sentenceSynsetData)):
                    if row[inRowIndex] not in synsetComparisonData:
                        synsetComparisonData[row[inRowIndex]] = [[row[0],
                            compareSynset(row[inRowIndex],
                            sentenceSynsetData[rowIndex][1:],simMeasureIndex)]]
                    else:
                        synsetComparisonData[row[inRowIndex]].append([row[0],
                            compareSynset(row[inRowIndex],
                            sentenceSynsetData[rowIndex][1:],simMeasureIndex)])
    return synsetComparisonData

# Chooses the sense with the highest overall similarity to any sense
def bestSimToAnySense(Data,synsetComparisonData):
    chosenSenses = []
    for wordRow in Data:
        rowLength = len(wordRow)
        bestSynset = None
        bestSynsetSim = 0
        currentSynsetData = None
        for synsetIndex in range(1,rowLength):
            currentSynsetData = synsetComparisonData.get(wordRow[synsetIndex])
            if currentSynsetData is not None:
                synsetDataEntries = len(currentSynsetData)
                for synsetDataEntry in range(0,synsetDataEntries):
                    synsetDataEntrySim = currentSynsetData[\
                                                       synsetDataEntry][1][1][0]
                    if(synsetDataEntrySim > bestSynsetSim):
                        bestSynset = wordRow[synsetIndex]
                        bestSynsetSim = synsetDataEntrySim
        chosenSenses.append(bestSynset)
    return chosenSenses
# Chooses the sense with the highest average similarity to all senses
def averageSimToAllSenses(Data,synsetComparisonData):
    chosenSenses = []
    for wordRow in Data:
        rowLength = len(wordRow)
        bestSynset = None
        bestSynsetAvgSim = 0
        currentSynsetData = None
        for synsetIndex in range(1, rowLength):
            currentSynsetData = synsetComparisonData.get(wordRow[synsetIndex])
            if currentSynsetData is not None:
                synsetDataEntries = len(currentSynsetData)
                synsetDataEntrySim = 0
                for synsetDataEntry in range(0, synsetDataEntries):
                    synsetDataEntrySim += currentSynsetData[\
                                                          synsetDataEntry][1][0]
                synsetDataEntrySim /= synsetDataEntries
                if(synsetDataEntrySim > bestSynsetAvgSim):
                    bestSynsetAvgSim = synsetDataEntrySim
                    bestSynset = wordRow[synsetIndex]
        chosenSenses.append(bestSynset)
    return chosenSenses
# Chooses the sense with... you got it.. the best similarity to the first sense
def bestSimToTopSense(Data,synsetComparisonData):
    chosenSenses = []
    for wordRow in Data:
        rowLength = len(wordRow)
        bestSynset = None
        bestSynsetSim = 0
        currentSynsetData = None
        for synsetIndex in range(1, rowLength):
            currentSynsetData = synsetComparisonData.get(wordRow[synsetIndex])
            if currentSynsetData is not None:
                synsetDataEntries = len(currentSynsetData)
                for synsetDataEntry in range(0, synsetDataEntries):
                    synsetDataEntrySim = currentSynsetData[\
                                                          synsetDataEntry][1][2]
                    if (synsetDataEntrySim > bestSynsetSim):
                        bestSynset = wordRow[synsetIndex]
                        bestSynsetSim = synsetDataEntrySim
        chosenSenses.append(bestSynset)
    return chosenSenses

#Chooses a random sense
def randomSense(Data,SynsetComparisonData):
    chosenSenses = []
    for wordRow in Data:
        chosenSenses.append(random.choice(wordRow[1:]))
    return chosenSenses


# Given a previous chosen sense, this returns the sense with the best similarity
#  to that sense
def bestSimToChosenSense(Data,synsetComparisonData,simMeasureIndex):
    chosenSenses = []
    previousChosenSenses = bestSimToTopSense(Data,synsetComparisonData)
    for wordRow in Data:
        rowLength = len(wordRow)
        bestSynset = None
        bestSynsetSim = 0
        currentSynsetData = None
        for synsetIndex in range(1, rowLength):
            currentSynsetData = synsetComparisonData.get(wordRow[synsetIndex])
            if currentSynsetData is not None:
                synsetDataEntries = len(currentSynsetData)
                for synsetDataEntry in range(0, synsetDataEntries):
                    # Needs Clarification, which synset do we compare against
                    # for every word?
                    synsetDataEntrySim = similarityMeasures[simMeasureIndex](
                        wordRow[synsetIndex],
                        previousChosenSenses[Data.index(wordRow)])
                    if (synsetDataEntrySim > bestSynsetSim):
                        bestSynset = wordRow[synsetIndex]
                        bestSynsetSim = synsetDataEntrySim
        chosenSenses.append(bestSynset)
    return chosenSenses

scoringTechniques = {0:bestSimToAnySense, 1:averageSimToAllSenses,
                     2:bestSimToTopSense,3:randomSense,4:bestSimToChosenSense}

# Performs the disambiguation
def pickBestSynsetPerWord(data,desiredScoringTechnique,simMeasureIndex,
                                                                    windowSize):
    synsetComparisonData = compareAllSynsets(windowSize, simMeasureIndex, data)
    if(desiredScoringTechnique <=3):
        return scoringTechniques[desiredScoringTechnique](data,
                                                        synsetComparisonData)
    else:
        return scoringTechniques[desiredScoringTechnique](data,
                                        synsetComparisonData,simMeasureIndex)
