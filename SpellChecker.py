import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity

# Kurang cukup kalau pakai n-gram overlapping technique

# Bakal Di coba Digabungin dengan :
# 1. Soundex Fusion


def dummy_fun(doc):
    return doc


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def loadWords(fileName):
    wordList = []
    file = open(fileName, 'r').read().splitlines()
    for line in file:
        wordList.append(line.lower())
    return wordList


def repeatcharNormalize(word):
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    for i in range(len(alphabet)):
        charac_long = 5
        while charac_long >= 3:
            char = alphabet[i]*charac_long
            word = word.replace(char, alphabet[i])
            charac_long -= 1
    return word


def preprocessing(tokenizedWords):
    # tokenizedWords = repeatcharNormalize(tokenizedWords)
    for index in range(len(tokenizedWords)):
        tokenizedWords[index] = repeatcharNormalize(tokenizedWords[index])
    return tokenizedWords


def makeBigramModels(words):
    tokenizedWords = []
    for word in words:
        tokenizedWord = list(word)
        tokenizedWord.insert(0, '<')
        tokenizedWord.extend('>')
        bigramCharacters = []
        for index in range(len(tokenizedWord)):
            if tokenizedWord[index] == '>':
                break
            elif tokenizedWord[index + 1] == '>':
                bigrams = [tokenizedWord[index] + '' + tokenizedWord[index + 1]]
            elif (tokenizedWord[index + 2] == '>') | (tokenizedWord[index] == '<'):
                bigrams = [tokenizedWord[index] + '' + tokenizedWord[index + 1]]
            else:
                bigrams = [(tokenizedWord[index] + '' + tokenizedWord[index + 1]),
                           (tokenizedWord[index] + '' + tokenizedWord[index + 2])]
            bigramCharacters.extend(bigrams)
        if ''.join(tokenizedWord[-4:-1]) == 'ang':
            bigramCharacters.extend([(tokenizedWord[-5] + '' + tokenizedWord[-2])])
        tokenizedWords.append(bigramCharacters)
    return tokenizedWords


def makeCountVector(tokenizedWords):
    countVectorizer = CountVectorizer(analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun) ## UNUSED, token_pattern=None)
    features = countVectorizer.fit_transform(tokenizedWords).toarray()

    tfidf = TfidfTransformer()
    features = tfidf.fit_transform(features).toarray()

    return features

if __name__ == '__main__':
    basicWords = loadWords('ivan-lanin-kata-dasar.txt')
    tokenizedBasicWords = makeBigramModels(basicWords)

    # typoWords = input("Masukkan Kata Typo : ")
    typoWords = loadWords('kata-typo.txt')
    preprocessedTypoWords = preprocessing(typoWords)
    # preprocessedTypoWords = preprocessing(typoWords)
    tokenizedTypoWords = makeBigramModels(preprocessedTypoWords)

    combinedTokenizedWords = tokenizedBasicWords + tokenizedTypoWords
    combinedWordsCountVector = makeCountVector(combinedTokenizedWords)

    basicWordsCountVector  = combinedWordsCountVector[0:(len(tokenizedBasicWords))]
    typoWordsCountVector  = combinedWordsCountVector[(len(tokenizedBasicWords)):]

    for index in range(len(typoWordsCountVector)):
        cosineSimilarityScores = []
        # jaccardSimilarityScores = []
        for index2 in range(len(basicWordsCountVector)):
            # jaccardSimilarityScores.append(jaccard_similarity_score(basicWordsCountVector[index2], typoWordsCountVector[index]))
            cosineSimilarityScores.append(cos_sim(basicWordsCountVector[index2], typoWordsCountVector[index]))

        # correctionIndex = jaccardSimilarityScores.index(max(jaccardSimilarityScores))
        cosineSimilarityScores = np.array(cosineSimilarityScores)
        correctionIndex = (-cosineSimilarityScores).argsort()[0:10]
        # jaccardSimilarityScores = np.array(jaccardSimilarityScores)
        # correctionIndex = (-jaccardSimilarityScores).argsort()[0:10]
        print("")
        print("Kata Typo : ", typoWords[index])
        print("Saran Pembenaran Kata : ")
        for index in range(len(correctionIndex)):
            print((index+1), basicWords[correctionIndex[index]], (cosineSimilarityScores[correctionIndex[index]])%100)