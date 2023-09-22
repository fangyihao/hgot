'''
Created on Sep. 7, 2023

@author: Yihao Fang
'''
import dsp
def _eval(s):
    try:
        r = eval(s)
        if isinstance(r, list):
            return r
        else:
            return [s]
    except:
        return [s]

def df_to_dsp(df):
    return [dsp.Example(question=row["Question"], answer=_eval(row["Answer"]), history=_eval(row["Context"])) if "Context" in row else dsp.Example(question=row["Question"], answer=_eval(row["Answer"])) for index, row in df.iterrows()]

def df_to_dsp_augmented(df):
    return [dsp.Example(question=row["Question"], context=_eval(row["Context"]), plan=row["Plan"], dependencies=row["Dependencies"], rewrite_questions=row["Rewrite Questions"], rewrite=row["Rewrite"], augmented=True) for index, row in df.iterrows()]
