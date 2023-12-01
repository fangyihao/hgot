'''
Created on Sep. 7, 2023

@author: Yihao Fang
'''
import dsp
def _eval(s):
    if isinstance(s, list):
        return s
    try:
        r = eval(s)
        if isinstance(r, list):
            return r
        else:
            return [s]
    except:
        return [s]

def df_to_dsp(df):
    return [dsp.Example(question=row["Question"], answer=_eval(row["Answer"]), history=_eval(row["Context"])) if "Context" in row 
            else dsp.Example(question=row["Question"], answer=row["Answer"], labels=_eval(row["Labels"])) if "Labels" in row
            else dsp.Example(question=row["Question"], answer=_eval(row["Answer"])) for index, row in df.iterrows()]

def df_to_dsp_augmented(df, segment):
    if segment == "plan":
        return [dsp.Example(question=row["Question"], context=_eval(row["Plan Context"]), plan=row["Plan"], dependencies=row["Dependencies"], augmented=True) for index, row in df.iterrows() 
                if row["Question"] is not None and row["Plan Context"] is not None and row["Plan"] is not None and row["Dependencies"] is not None]
    elif segment == "rewrite":
        return [dsp.Example(question=row["Question"], rewrite_context=row["Rewrite Context"], rewrite=row["Rewrite"], augmented=True) for index, row in df.iterrows() 
                if row["Question"] is not None and row["Rewrite Context"] is not None and row["Rewrite"] is not None]
    elif segment == "rationale":
        return [dsp.Example(question=row["Question"], context=_eval(row["Rationale Context"]), rationale = row["Rationale"], answer = row["Answer"], augmented=True) for index, row in df.iterrows()
                if row["Question"] is not None and row["Rationale Context"] is not None and row["Rationale"] is not None and row["Answer"] is not None]
    else:
        raise NotImplementedError()