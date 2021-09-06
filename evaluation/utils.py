import json
def load(infile):
    with open(infile, 'r') as df:
        f = json.load(df)
    return f

def dump(payload, outfile):
    with open(outfile, 'w') as df:
        f = json.dump(payload, df, indent=4)
    return f