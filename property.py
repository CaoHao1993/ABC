import pickle

class Property(object):
    def __init__(self, vocabulary, inverse_vocabulary, max_seq_length):
        self.vocabulary = vocabulary
        self.inverse_vocabulary = inverse_vocabulary
        self.max_seq_length = max_seq_length
    
def save_property(p, property_path):
    f = open(property_path,'wb')
    pickle.dump(p, f)
    f.close()
    
def load_property(property_path):
    f = open(property_path,'rb')
    p = pickle.load(f)
    return p