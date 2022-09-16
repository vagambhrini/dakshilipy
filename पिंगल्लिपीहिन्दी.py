#!/usr/bin/env python
# coding: utf-8

# In[13]:


from ideas import import_hook, utils
import tokenize as py_tokenize
import token_utils

def दर्शय(*args):
    res = []
    for arg in args:
        if type(arg) == type(True) and arg == True:
            res.append("सत")
        elif type(arg) == type(True) and arg == False:
            res.append("असत")
        elif type(arg) == type(None) and arg == None:
            res.append("अज्ञात")
        else:
            res.append(arg)
    print(*res)

# = map
श्रेणी = list
क्रमश्रेणी = range
अधोमुख = reversed

hi_to_py = {
    "असत": "False",
    "अज्ञात": "None",
    "सत": "True",
    "च": "and",
    "यथा": "as",
    "वा": "assert",
    "सर्ग": "def",
    "यदि": "if",
    "अन्यथा": "else",    
    "आनय": "import",      
    "नहीं": "not",
    "या": "or",  
    "सर्गफल": "return",  
    "अपसरण": "break",
    "क्रमशः": "for",
    "में": "in",
    "यदा": "while",
    "समीकरण": "lambda",
    "अध्यास": "map",
    "पादसंख्या": "len",
    # अवधी: : range
    #"अनुव्रत्ति": "for"
    #"पुनराव्रत्ति": "for"
    # a few builtins useful for beginners    
    #"दर्शय": "print",
    #"श्रेणि": "range",
    #"async": "async",  # do not translate
    #"await": "await",  # as these are not for beginners
    #"is": "is",    
    #"lambda": "lambda",
    #"nonlocal": "nonlocal",
    #"pass": "pass",
    #"raise": "raise",    
    #"from": "from",
    #"global": "global",    
    #"except": "except",
    #"finally": "finally",    
    #"del": "del",
    #"elif": "elif",    
    #"class": "class",
    #"continue": "continue",    
    #"try": "try",
    #"while": "while",
    #"with": "with",
    #"yield": "yield",
    # a few builtins useful for beginners
    #"input": "input",
    #"exit": "exit",  # useful for console
}

hi_to_py_tokens = [tuple(([x.string for x in token_utils.tokenize(token) if x.string != ''], val)) for (token, val) in hi_to_py.items()]
hi_to_py_first = {}

for (i, (hi, py)) in enumerate(hi_to_py_tokens):
    if hi[0] not in hi_to_py_first:
        hi_to_py_first[hi[0]] = []
    hi_to_py_first[hi[0]].append((len(hi), i))
for k, v in hi_to_py_first.items():
    v.sort(reverse=True)


def untokenize(tokens):
    """Return source code based on tokens.
    Adapted from https://github.com/myint/untokenize,
    Copyright (C) 2013-2018 Steven Myint, MIT License (same as this project).
    This is similar to Python's own tokenize.untokenize(), except that it
    preserves spacing between tokens, by using the line
    information recorded by Python's tokenize.generate_tokens.
    As a result, if the original soure code had multiple spaces between
    some tokens or if escaped newlines were used or if tab characters
    were present in the original source, those will also be present
    in the source code produced by untokenize.
    Thus ``source == untokenize(tokenize(source))``.
    Note: if you you modifying tokens from an original source:
    Instead of full token object, ``untokenize`` will accept simple
    strings; however, it will only insert them *as is* without taking them
    into account when it comes with figuring out spacing between tokens.
    """
    words = []
    previous_line = ""
    last_row = 0
    last_column = -1
    last_non_whitespace_token_type = None

    for token in tokens:
        if isinstance(token, str):
            words.append(token)
            continue
        if token.type == py_tokenize.ENCODING:
            continue

        # Preserve escaped newlines.
        if (
            last_non_whitespace_token_type != py_tokenize.COMMENT
            and token.start_row > last_row
        ):
            if previous_line.endswith(("\\\n", "\\\r\n", "\\\r")):
                #pass
                words.append(previous_line[len(previous_line.rstrip(" \t\n\r\\")) :])

        # Preserve spacing.
        if token.start_row > last_row:
            last_column = 0
        if token.start_col > last_column:
            whitelist = set(' \\\n \\\r \\\r\n')
            line_white = ''.join(filter(whitelist.__contains__, token.line[last_column : token.start_col]))
            words.append(line_white)
            #words.append(token.line[last_column : token.start_col])

        words.append(token.string)

        previous_line = token.line
        last_row = token.end_row
        last_column = token.end_col
        if not token.is_space():
            last_non_whitespace_token_type = token.type

    return "".join(words)
    
def match(stack, tokens):
    #print("matching: ", stack, " *** ", tokens)
    res = False
    for s, t in zip(stack, tokens):
        res = True # if len > 0: res = True
        if s.string != t: return False
    return res
    
def matchIndices(stack, indices):
    for i, index in enumerate(indices):
        if match(stack, hi_to_py_tokens[index[1]][0]):
            return i
    return None

def collapse(curStack, indices, new_tokens):
    matchIndex = matchIndices(curStack, indices)
    #print("eka: ", new_tokens)
    if matchIndex == None:
        new_tokens.extend(curStack)
    else:
        #for x in curStack:
        #    x.string =
        #for i in indices[matchIndex][0]:   
        curStack[0].string = hi_to_py_tokens[indices[matchIndex][1]][1]
        #curStack[0].line = hi_to_py_tokens[indices[matchIndex][1]][0][0]
        curStack[0].end = curStack[-1].end
        new_tokens.append(curStack[0])
        #new_tokens.append(hi_to_py_tokens[indices[matchIndex][1]][1])
        #print("dwa: ", new_tokens)
        #print("putting: ", hi_to_py_tokens[indices[matchIndex][1]][1] , curStack[indices[matchIndex][0]:])
        #print("info: ", indices[matchIndex][0])
        new_tokens.extend(curStack[indices[matchIndex][0]:])
        #print("trini: ", new_tokens)
    
def transform_source१(source, **_kwargs):
    """A simple replacement of 'French Python keyword' by their normal
    English version.
    """
    new_tokens = []
    matches = []
    curMatch = 0
    indices = None
    tokens = token_utils.tokenize(source)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        pretoken = tokens[i-1] if i > 0 else None
        if token.string in hi_to_py_first and (type(pretoken) == type(None) or pretoken.end != token.start or (pretoken.type != 1 and pretoken.type != 60)):
            indices = hi_to_py_first[token.string]
            maxLen = indices[0][0]
            curMatch = min(i+maxLen, len(tokens))
            matchIndex = matchIndices(tokens[i:curMatch], indices)
            #print("checking: ", matchIndex, tokens[i:curMatch])
            if matchIndex == None:
                i += 1
            else:
                token.string = hi_to_py_tokens[indices[matchIndex][1]][1]
                token.end = tokens[i+indices[matchIndex][0]].end
                i += indices[matchIndex][0]
                #print("adding transformed token: ", token)
            #matchLen = collapse(tokens[i:curMatch], indices, new_tokens)
            #print("ardhaadhik trini: ", token, new_tokens)
            #val = hi_to_py_tokens[index]
            #token.string = hi_to_py[token.string]
        else:
            i += 1
            #print("chatwari: ", token, new_tokens)
            #print("adding: ", token)
        new_tokens.append(token)
            #print("panch: ", token, new_tokens)
    #new_source = token_utils.untokenize(new_tokens)
    new_source = untokenize(new_tokens)
    #print("final: ", new_source, new_tokens)
    return new_source

def add_hook१(**_kwargs):
    """Creates and adds the import hook in sys.meta_path.
    Uses a custom extension for the exception hook."""
    hook = import_hook.create_hook(
        transform_source=transform_source१,
        hook_name='sanskrit',
        extensions=[".pyhi"],
    )
    return hook


# In[6]:


class VistarSyntaxError(Exception):
    """Currently, only raised when a repeat statement has a missing colon."""

    pass

def getTokensList(tokens):
    li = []
    for token in tokens:
        if token.is_space() or token.is_comment():
            continue
        li.append(token)
    return li

def searchString(tokens, s):
    for i, token in enumerate(tokens):
        if token.string == s: return i
    return None

def transform_source२(source, **_kwargs):
    token = "क्रमशः"
    start = [x.string for x in token_utils.tokenize(token) if x.string != '']
    token = "का विस्तार"
    end = [x.string for x in token_utils.tokenize(token) if x.string != '']
    new_tokens = []
    # यदायदा क तदा ख :
    # क का ख में विस्तार :
    # क्रमशः क में ख का विस्तार :
    for tokens in token_utils.get_lines(source):
        # a line of tokens can start with INDENT or DEDENT tokens ...
        tli= getTokensList(tokens)
        #print("orig: ", tokens)
        #ntli = []
        if match(tli, start) and tli[4].end != tli[5].start:
            colonIndex = searchString(tli, ":")
            if colonIndex == None:
                raise VistarSyntaxError(
                    "विस्तार के लिये अंत में : का उपयोग करे|"
                    + f"{tli[0].start_row}\n    {tli[0].line}."
                )
            if match(reversed(tli[:colonIndex]), reversed(end)):
                colonIndex = searchString(tokens, ":")
                vistarIndexRel = searchString(reversed(tokens[:colonIndex]), "क")
                vistarIndex = len(tokens[:colonIndex]) - vistarIndexRel
                #print ("?? this: ", colonIndex, vistarIndexRel, vistarIndex)
                tokens = tokens[:vistarIndex-1] + tokens[colonIndex:]
            #print("tokens: ", tokens)
            #print("joined: ", untokenize(tokens))
            #repeat_index = token_utils.get_first_index(tokens)
            #second_token = tokens[repeat_index + 1]
            #first_token.string = "for %s in %s" % tli[1], tli[3], 
            #last_token.string = "):"
        new_tokens.extend(tokens)
    return untokenize(new_tokens)


def add_hook२(predictable_names=False, **_kwargs):
    """Creates and adds the import hook in sys.meta_path.
    If ``predictable_names`` is set to ``True``, a callback parameter
    passed to the source transformation function will be used to
    create variable loops with predictable names."""
    callback_params = {"predictable_names": predictable_names}
    hook = import_hook.create_hook(
        transform_source=transform_source२,
        callback_params=callback_params,
        hook_name=__name__,
    )
    return hook

if __name__ != '__main__':
    add_hook२()
    add_hook१()
