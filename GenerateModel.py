from nltk import CRFTagger

if __name__ == '__main__':
    filename = "Indonesian_Manually_Tagged_Corpus.tsv"
    with open(filename, "r", encoding="utf-8") as f:
        datas = f.read().split("\n\n")
    taggedkalimat = []
    for data in datas[:len(datas)]:
        kalimat = data.split("\n")
        taggedkata = []
        for kata in kalimat[:]:
            k, tag = kata.split("\t")
            kata_tag = (k, tag)
            taggedkata.append(kata_tag)
        taggedkalimat.append(taggedkata)
    ctagger = CRFTagger()
    modelname = input("Save model as : ")
    ctagger.train(taggedkalimat, modelname)
    print("Generated model from %s into %s" % (filename, modelname))
    print("Usage :")
    print("\t\tnltk.CRFTagger().set_model_file(your_model)")
    print("\t\tnltk.CRFTagger().tag_sents(list(your_sentence))")
