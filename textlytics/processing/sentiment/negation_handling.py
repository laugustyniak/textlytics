# coding: utf-8

import re as _re

_full_negation = "not, no, none, never, nothing, nobody, nowhere, neither, nor"
_quasi_negation = "hardly, scarcely"
_abs_negation = "not at all, by no means, in no way, nothing short of"
_negation = _full_negation+", "+_quasi_negation+", "+_abs_negation
_negation_regexp = "(?i)(" + (")|(".join( _negation.split(", ") )) + ")"

_quasi_negatives = "not every, not all, not much, not many, not always, not never"
_quasi_negatives = "(?i)(" + (")|(".join( _quasi_negatives.split(", ") )) + ")"

_interpunction = ", . ;".split(" ")

def _filterEmpty(x):
    """x is an iterable (e.g. list)
    function assumes that each element in the iterable is a string and is passes only non-empty strings
    """
    new = []
    for i in x:
        if i and i != "not":
            new.append( i )
    return new

def handle_negation( sentence ):
    """ s is a sentence written in english
    This function tries to add "not_" prefix to all of the words, that are negated in that sentence
    """
    parsed = _re.sub(_quasi_negatives, "", sentence) #remove 
    parsed = _re.sub(_negation_regexp, "not", parsed) # assume all negation words mean the same as not
    parsed = _re.sub("(?i)(\w+)n't", "\\1 not", parsed) #change n't to not
    parsed = _re.sub("([.,;:])", " \\1", parsed) # add additional space to enable later split
    tokens = _re.split("[ ]", parsed)
#     print tokens
    
    flag = False
    for i in range(len(tokens)):
        if flag:
            if tokens[i] in _interpunction:
                flag = False
            else:
                tokens[i] = "not_"+tokens[i]
        else:
            if _re.match(_negation_regexp, tokens[i]):
                flag = True

    tokens = _filterEmpty( tokens )

    return " ".join(tokens)
