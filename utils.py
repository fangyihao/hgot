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

def df_to_dsp_augmented(df, segment):
    if segment == "plan":
        return [dsp.Example(question=row["Question"], context=_eval(row["Plan Context"]), plan=row["Plan"], dependencies=row["Dependencies"], augmented=True) for index, row in df.iterrows()]
    elif segment == "rewrite":
        return [dsp.Example(question=row["Question"], rewrite_context=row["Rewrite Context"], rewrite=row["Rewrite"], augmented=True) for index, row in df.iterrows()]
    elif segment == "rationale":
        return [dsp.Example(question=row["Question"], context=_eval(row["Rationale Context"]), rationale = row["Rationale"], answer = row["Answer"], augmented=True) for index, row in df.iterrows()]
    else:
        raise NotImplementedError()