class Tokenizer:
    def __init__(self,vocab_size=500):
        self.vocab_size=vocab_size
        self.general_vocab=256
        self.sos="<|SOS|>"
        self.eos="<|EOS|>"
        self.ukn="<|UKN|>"
        self.enc_eos=vocab_size-1
        self.enc_sos=vocab_size-2
        self.enc_ukn=vocab_size-3
        self.merges={}
        self.vocab={}
    
    def _to_token(self,text)->list:
        tokens=[]
        for sen in text:
            toks=list(map(int,sen.encode('utf-8')))
            toks=[self.enc_sos] + toks + [self.enc_eos]
            for t in toks:
                tokens.append(t)
        return tokens
    def _get_stats(self,ids):
        frequency={}
        for p in zip(ids,ids[1:]):
            frequency[p] = frequency.get(p,0) + 1
        return frequency

    def _merge(self,ids,pair,idx):
        new_ids=[]
        i=0
        while i<len(ids):
            if i<len(ids)-1 and ids[i]==pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i+=2
            else:
                new_ids.append(ids[i])
                i+=1
        return new_ids

    def train(self,text):
        tokens=self._to_token(text)
        for i in range(self.vocab_size - self.general_vocab-1):
            idx=self.general_vocab + i
            stats=self._get_stats(tokens)
            pair=max(stats,key=stats.get)
            tokens=self._merge(tokens,pair,idx)
            self.merges[pair] = idx
        self.vocab={idx : bytes([idx]) for idx in range(self.general_vocab)}
        self.vocab[self.enc_sos] = bytes(self.sos.encode('utf-8'))
        self.vocab[self.enc_eos] = bytes(self.eos.encode('utf-8'))
        self.vocab[self.enc_ukn] = bytes(self.ukn.encode('utf-8'))
        for (p0,p1),enc in (self.merges.items()):
            self.vocab[enc] = self.vocab[p0] + self.vocab[p1]
    def decode(self,tokens):
        result = []
        for t in tokens:
            if t == self.enc_sos:
                result.append(self.sos)
            elif t == self.enc_eos:
                result.append(self.eos)
            elif t == self.enc_ukn:
                result.append(self.ukn)
            elif t in self.vocab:
                try:
                    result.append(self.vocab[t].decode('utf-8'))
                except Exception:
                    result.append(self.ukn)
            else:
                result.append(self.ukn)
        return ''.join(result)
    def encode(self,text):
        tokens=list(map(int,text.encode('utf-8')))
        tokens=[self.enc_sos] + tokens + [self.enc_eos]
        while len(tokens)>1:
            stats=self._get_stats(tokens)
            pair=min(stats,key=lambda p:self.merges.get(p,float('inf')))
            if pair not in self.merges:
                break
            idx=self.merges.get(pair,self.enc_ukn)
            tokens=self._merge(tokens,pair,idx)
        tokens=[t if t in self.vocab else self.enc_ukn for t in tokens]
        return tokens