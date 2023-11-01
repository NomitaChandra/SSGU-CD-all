import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("The rats in the control group were intraperitoneally ( i . p . ) injected with 0 . 9 % saline ( 4 ml / kg ) ; the rats in the model group were i . p .")
for token in doc:
    print('{0}({1}) <-- {2} -- {3}({4})'.format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))
# 依存句法树打印输出

displacy.serve(doc, style='dep')
