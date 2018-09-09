# The purpose of this module is to convert nltk wordnet from a hierarchy into a graph in preparation for the
# use of the Personalised Pagerank Algorithm
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np


tokenizer = RegexpTokenizer(r'\w+')
stopWords = set(stopwords.words('english'))

# Class is used to store a synset and its meaningful definition words along with their synsets, effectively showing the
# synsets it is linked to, this will be the link in the graph
class SynsetDetails:

    def __init__(self, synset):
        self.synset = synset
        self.definitionSynsets = self.getDefinitionSynsets(
                                            self.getFilteredTokenizedDefinition(
                                                tokenizer.tokenize(synset.definition())))

    # Takes a sentence as input, returns the sentence with stopwords stripped
    # Stopwords are words such as I, The, And etc.
    def getFilteredTokenizedDefinition(self, definition):
        filteredTokenizedDefinition = []
        for word in definition:
            if word not in stopWords:
                filteredTokenizedDefinition.append(word)
        return filteredTokenizedDefinition

    # The synsets of every word in the definition input sentence are extracted
    # and stored in a dictionary for later access with the word as a key.
    def getDefinitionSynsets(self, filteredTokenisedDefinition):
        definitionSynsets = dict()
        for word in filteredTokenisedDefinition:
            wordsSynsets = wordnet.synsets(word)
            if wordsSynsets != []:
                definitionSynsets[word] = wordsSynsets
        return definitionSynsets

# Makes a list of wordnets synset definition links, stores them in a list, and
# keeps track of which synset they reference.
def getWordnetSynsetRelations():
    allSynsets = []
    count = 0
    synsetIndex = dict()
    for synset in wordnet.all_synsets():
        currentSynsetDetails = SynsetDetails(synset)
        allSynsets.append(currentSynsetDetails)
        synsetIndex[(synset.pos(),synset.offset())] = count
        count+=1
    return [allSynsets, synsetIndex]

# Generates the data needed to represent wordnet as a graph, using definition
# relations as edges, along with wordnet hierarchy relations between synsets
# such as Hyponym, Holonym etc
def generateAndSaveWordnetGraph():
    wordnetSynsetRelations = getWordnetSynsetRelations()
    wordnetSynsetIndexes = wordnetSynsetRelations[1]
    row = []
    col = []
    data = []
    for synsetEntry in wordnetSynsetRelations[0]:
        # Get the definition links
        synsetLinks = list(synsetEntry.definitionSynsets.values())
        # Add other links
        synsetLinks.append(synsetEntry.synset.hypernyms())
        synsetLinks.append(synsetEntry.synset.hyponyms())
        synsetLinks.append(synsetEntry.synset.member_holonyms())

        # Gather all the synsets this synset is in some way related to in one list
        synsetLinksUnion = []
        for listOfSynsets in synsetLinks:
            synsetLinksUnion = synsetLinksUnion + listOfSynsets

        # Convert set to remove duplicates
        synsetLinksUnion = set(synsetLinksUnion)
        for linkedToSynset in synsetLinksUnion:
            col.append(wordnetSynsetIndexes.get((synsetEntry.synset.pos(),synsetEntry.synset.offset())))
            row.append(wordnetSynsetIndexes.get((linkedToSynset.pos(),linkedToSynset.offset())))
            data.append(1)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    np.savez("initialGraphData",row=row,col=col,data=data,synsetIndexes=wordnetSynsetIndexes)
    return None
generateAndSaveWordnetGraph()
