import json
from collections import defaultdict
import pandas as pd

def stats_bios():
    data = [json.loads(item) for item in open('data/bias-bios/test.jsonl','r').readlines()]
    occ = defaultdict(lambda : {'m' : 0, 'f' : 0})
    for i in range(len(data)):
        occ[data[i]['p']][data[i]['g']] += 1
    
    df = pd.DataFrame(occ).T
    df.to_csv('bios_stats.csv')



def stats_gab():
    data = [json.loads(item) for item in open('data/gab/train.jsonl','r').readlines()]

    data_attr = ["lesbian","gay", "bisexual", "transgender", "trans", "queer", "lgbt", "lgbtq", "homosexual", "straight", "heterosexual", "male", "female",
        "nonbinary", "african", "african american", "black", "white", "european", "hispanic", "latino", "latina", "latinx", "mexican", "canadian", "american",
        "asian", "indian", "middle eastern", "chinese", "japanese", "christian", "muslim", "jewish", "buddhist", "catholic", "protestant", "sikh", "taoist", 
        "old", "older", "young", "younger", "teenage", "millenial", "middle aged", "elderly", "blind", "deaf", "paralyzed"]
    
    group = defaultdict(lambda : [])

    for i in range(len(data)):
        for attr in data_attr:
            if attr in data[i]['text'] :                                
                group[attr].append(data[i]['label'])


    for k in group:
        group[k] = sum(group[k]) / (len(group[k])+ 0.0001)
        print (k , group[k])




    import pdb
    pdb.set_trace()



# stats_gab ()
stats_bios()    