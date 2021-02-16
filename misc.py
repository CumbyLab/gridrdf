

def int_or_str(value):
    '''
    For augment parser trim, where input can be either integer or string
    '''
    try:
        return int(value)
    except:
        return value


def read_and_merge_similarity_matrix():
    import pandas as pd

    t = pd.DataFrame([])
    for i in range(0, 12000, 100):
        a = pd.read_csv(str(i)+'_'+str(i+100)+'.csv', index_col=0)
        t = pd.concat([t,a])

    a = pd.read_csv(str(12000)+'_'+str(12177)+'.csv', index_col=0)
    t = pd.concat([t,a])

    t.to_csv('total_matrix.csv')