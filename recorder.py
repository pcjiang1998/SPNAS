class Recorder:
    def __init__(self, fingerprint, original_encode, data, simplified_encode=None, tmp_encode=None):
        self.fingerprint = fingerprint
        self.original_encode = original_encode
        self.simplified_encode = simplified_encode
        self.tmp_encode = tmp_encode
        self.data = data

    def __hash__(self):
        return hash(self.fingerprint)

    def __eq__(self, other):
        if isinstance(other, Recorder):
            return self.fingerprint == other.fingerprint
        else:
            return False
