

def int_or_str(value):
    '''
    For augment parser trim, where input can be either integer or string
    '''
    try:
        return int(value)
    except:
        return value
