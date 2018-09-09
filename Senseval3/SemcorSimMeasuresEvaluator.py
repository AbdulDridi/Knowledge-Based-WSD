import nltk
import sys
sys.path.insert(0, '../SimilarityMeasures')
from SimPythonLexicalDisambiguator import *

#load all data needed into the relevant data structures
corpus1 = nltk.corpus.reader.semcor.SemcorCorpusReader(".", "d000.xml",
                                                                   nltk.wordnet)
corpus2 = nltk.corpus.reader.semcor.SemcorCorpusReader(".", "d001.xml",
                                                                   nltk.wordnet)
corpus3 = nltk.corpus.reader.semcor.SemcorCorpusReader(".", "d002.xml",
                                                                   nltk.wordnet)
#Append all into one sentence
sentences = list(corpus1.sents())
for s in corpus2.sents():
    sentences.append(s)
for s in corpus3.sents():
    sentences.append(s)

sentencesSynsets = list()
sentencesMeaningfulSynsets = []

# Parse sentences and perform disambiguation
for s in range(len(sentences)):
    setOfSynsets = parseTextToValuableSynsetsAsList(sentences[s])
    sentencesSynsets.append(setOfSynsets[0])
    sentencesMeaningfulSynsets = setOfSynsets[1]
    if(setOfSynsets != []):
        #SECTION THAT DOES DISAMBIGUATION
        disambiguatedSentence = pickBestSynsetPerWord(
                                            sentencesMeaningfulSynsets, 3, 0, 2)
        for wordNumber in range(len(sentencesSynsets[s])):
            for synset in disambiguatedSentence:
                if sentencesSynsets[s][wordNumber]:
                    sentencesSynsets[s][wordNumber] = synset
                    disambiguatedSentence.remove(synset)
                    break
        #print(sentencesSynsets[s])

taggedSentences = list(corpus1.tagged_sents("d000.xml","both"))
for s in corpus2.tagged_sents("d001.xml","both"):
    taggedSentences.append(s)
for s in corpus3.tagged_sents("d002.xml","both"):
    taggedSentences.append(s)

aCorrect,nCorrect,vCorrect,adjCorrect,advCorrect = (0,0,0,0,0)
aIncorrect,nIncorrect,vIncorrect,adjIncorrect,advIncorrect = (0,0,0,0,0)
aMissed,nMissed,vMissed,adjMissed,advMissed = (0,0,0,0,0)

for taggedSentenceIndex in range(len(taggedSentences)):
    for wordIndex in range(len(taggedSentences[taggedSentenceIndex])):
        #if this contains the solution (i.e. we're testing it) and the solution
        # is valid this could be buggy
        if (taggedSentences[taggedSentenceIndex][wordIndex].height() == 3 and
                "None" not in taggedSentences[taggedSentenceIndex][wordIndex].
                                                                       label()):
            pos = taggedSentences[taggedSentenceIndex][wordIndex].label()\
                                                                  .split(".")[1]
            if(sentencesSynsets[taggedSentenceIndex][wordIndex] is not
                                                               (None or False)):
                if(str(sentencesSynsets[taggedSentenceIndex][wordIndex])[8:-2]
                       == taggedSentences[taggedSentenceIndex][wordIndex].
                                                                       label()):
                    aCorrect = aCorrect + 1
                    if(pos == "r"):
                        advCorrect = advCorrect + 1
                    elif(pos == "v"):
                        vCorrect = vCorrect + 1
                    elif(pos == "n"):
                        nCorrect = nCorrect + 1
                    elif(pos == "a"):
                        adjCorrect = adjCorrect + 1
                else:
                    aIncorrect = aIncorrect + 1
                    if (pos == "r"):
                        advIncorrect = advIncorrect  + 1
                    elif (pos == "v"):
                        vIncorrect = vIncorrect + 1
                    elif (pos == "n"):
                        nIncorrect = nIncorrect + 1
                    elif (pos == "a"):
                        adjIncorrect = adjIncorrect + 1
            elif(sentencesSynsets[taggedSentenceIndex][wordIndex] is None):
                aMissed = aMissed + 1
                if (pos == "r"):
                    advMissed = advMissed + 1
                elif (pos == "v"):
                    vMissed = vMissed + 1
                elif (pos == "n"):
                    nMissed = nMissed + 1
                elif (pos == "a"):
                    adjMissed = adjMissed + 1

precision = aCorrect / (aCorrect + aIncorrect)
recall = aCorrect / (aCorrect + aIncorrect + aMissed)
print("Correct:", aCorrect)
print("Incorrect:", aIncorrect)
print("Didn't Disambiguate:", aMissed)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", 2 * ((precision * recall) / (precision + recall)))
precision = adjCorrect / (adjCorrect + adjIncorrect)
recall = adjCorrect / (adjCorrect + adjIncorrect + adjMissed)
print("Adj F1:", 2 * ((precision * recall) / (precision + recall)))
precision = vCorrect / (vCorrect + vIncorrect)
recall = vCorrect / (vCorrect + vIncorrect + vMissed)
print("V F1:", 2 * ((precision * recall) / (precision + recall)))
precision = nCorrect / (nCorrect + nIncorrect)
recall = nCorrect / (nCorrect + nIncorrect + nMissed)
print("N F1:", 2 * ((precision * recall) / (precision + recall)))
precision = advCorrect / (advCorrect + advIncorrect)
recall = advCorrect / (advCorrect + advIncorrect + advMissed)
if(precision or recall > 0):
    print("Adv F1:", 2 * ((precision * recall) / (precision + recall)))
else:
    print("Adv F1: N/A (Divide by Zero)")
