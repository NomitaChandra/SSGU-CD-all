# genia-tagger-py
Python wrapper for [GENIA tagger](http://www.nactem.ac.uk/tsujii/GENIA/tagger/)

## Example usage


```python
import os
from geniatagger import GENIATagger

executable_path = os.path.join(".", "geniatagger-3.0.2", "geniatagger")
tagger = GENIATagger(executable_path)

text = "Recombinant MIP-1-alpha induces a dose-dependent inhibition of different strains of \
        HIV-1, HIV-2, and simian immunodeficiency virus (SIV)."

for word, base_form, pos_tag, chunk, named_entity in tagger.tag(text):
    print("{:20}{:20}{:10}{:10}{:10}".format(word, base_form, pos_tag, chunk, named_entity))

```

    Recombinant         Recombinant         JJ        B-NP      B-protein 
    MIP-1-alpha         MIP-1-alpha         NN        I-NP      I-protein 
    induces             induce              VBZ       B-VP      O         
    a                   a                   DT        B-NP      O         
    dose-dependent      dose-dependent      JJ        I-NP      O         
    inhibition          inhibition          NN        I-NP      O         
    of                  of                  IN        B-PP      O         
    different           different           JJ        B-NP      O         
    strains             strain              NNS       I-NP      O         
    of                  of                  IN        B-PP      O         
    HIV-1               HIV-1               NN        B-NP      O         
    ,                   ,                   ,         O         O         
    HIV-2               HIV-2               NN        B-NP      O         
    ,                   ,                   ,         O         O         
    and                 and                 CC        O         O         
    simian              simian              JJ        B-NP      O         
    immunodeficiency    immunodeficiency    NN        I-NP      O         
    virus               virus               NN        I-NP      O         
    (                   (                   (         O         O         
    SIV                 SIV                 NN        B-NP      O         
    )                   )                   )         O         O         
    .                   .                   .         O         O         

