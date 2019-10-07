from sklearn.metrics import f1_score
from fastai.text import *

@np_func
def f1(inp,targ): return f1_score(targ, np.argmax(inp, axis=-1), average='macro')