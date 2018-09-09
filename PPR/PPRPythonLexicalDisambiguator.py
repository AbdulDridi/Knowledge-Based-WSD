from nltk.corpus import wordnet
import numpy as np
from nltk.corpus import stopwords
from scipy.sparse import csc_matrix
from sklearn.preprocessing import normalize

# Creates a sparse matrix representation of a graph
def makeWordnetGraphAndLoadSynsetIndexDict():
    graphDataFile = np.load("../PPR/"
                            "initialGraphData.npz")
    synsetIndexes = graphDataFile['synsetIndexes'].item()
    length = len(synsetIndexes.keys())
    data = graphDataFile['data']
    row = graphDataFile['row']
    col = graphDataFile['col']
    wordnetAdjacencyMatrixGraph = csc_matrix((data, (row, col)),
                                            shape=(length, length))
    return (wordnetAdjacencyMatrixGraph, synsetIndexes)

# Creates transition matrix for Pagerank operations
def makeTransitionMatrix(adjacencyMatrix):
    wordnetTransitionMatrix = csc_matrix.copy(adjacencyMatrix)
    wordnetTransitionMatrix = normalize(wordnetTransitionMatrix,
                                        norm='l1', axis=1)
    return wordnetTransitionMatrix

# Returns the tokenized processed input text (assumed to be a sentence), along
# with the synsets corresponding to each of those words
def getSetSynsets(text):
    tokenizedText = text
    stopWords = set(stopwords.words('english'))
    stopWords.add("I")
    wordsWithSynsets = list()
    # This holds a list that either has the corresponding set of synsets for the
    # word in the input text or None if none found
    sentence = list()
    for word in tokenizedText:
        #if word not in stopWords:
        synsets = wordnet.synsets(word)
        if synsets != []:
            wordsWithSynsets.append(synsets)
            sentence.append(True)
        else:
            sentence.append(False)
    return (sentence,wordsWithSynsets)
# Distributes the initial rank in the graph to the synsets relevant to our
# sentence
def getInitialRankDistribution(textSynsets, noOfNodesInGraph, synsetIndexes):
    if(len(textSynsets)!= 0):
        initialRankDistribution = np.zeros(noOfNodesInGraph)
        rankSplitBetweenWords = 1/len(textSynsets)
        rankPerSynset = 0
        for synsetList in textSynsets:
            rankPerSynset = rankSplitBetweenWords / (len(synsetList))
            for synset in synsetList:
                pos = synset.pos()
                offset = synset.offset()
                nodeIndex = synsetIndexes.get((pos,offset))
                initialRankDistribution[nodeIndex] += rankPerSynset
    else:
        initialRankDistribution = np.full(noOfNodesInGraph, 1/noOfNodesInGraph)
    return initialRankDistribution

# For each word, the synset with the highest rank is selected as the
# correct disambiguated word.
def getTextSynsets(textSynsets ,pageRankVector, synsetIndexes):
    disambiguatedSynsets = []
    for wordSynsets in textSynsets:
        currentSynset = None
        currentSynsetRank = 0
        for synset in wordSynsets:
            pos = synset.pos()
            offset = synset.offset()
            synsetPagerankValue = pageRankVector[
                                                synsetIndexes.get((pos,offset))]
            if synsetPagerankValue > currentSynsetRank:
                currentSynset = synset
                currentSynsetRank = synsetPagerankValue
        disambiguatedSynsets.append(currentSynset)
    return disambiguatedSynsets

# Performs the whole disambiguation operation given pure text as input
def performPPRAndGetSynsets(text,transitionMatrix,synsetIndexes,dampingFactor,
                                                                    iterations):
    textSynsets = getSetSynsets(text)
    initialRankDistribution = getInitialRankDistribution(textSynsets,
                                transitionMatrix.get_shape()[0],synsetIndexes)
    pageRankVector = initialRankDistribution.copy()
    for i in range(iterations):
        pageRankVector = (dampingFactor
                          * (transitionMatrix.dot(pageRankVector)))\
                          + (1-dampingFactor) * initialRankDistribution

    return getTextSynsets(textSynsets,pageRankVector,synsetIndexes)
# Performs the whole disambiguation operation given the text synsets as input
# (Useful for senseval tasks)
def performPPRAndGetSynsetsFromSynsets(textSynsets,transitionMatrix,
                                        synsetIndexes,dampingFactor,iterations):
    initialRankDistribution = getInitialRankDistribution(textSynsets,
                                transitionMatrix.get_shape()[0],synsetIndexes)
    pageRankVector = initialRankDistribution.copy()
    for i in range(iterations):
        pageRankVector = (dampingFactor
                          * (transitionMatrix.dot(pageRankVector)))\
                          + (1-dampingFactor) * initialRankDistribution

    return getTextSynsets(textSynsets,pageRankVector,synsetIndexes)
