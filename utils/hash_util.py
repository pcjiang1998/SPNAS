import hashlib

__always_supported = ('md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512', 'blake2b', 'blake2s',
                      'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512', 'shake_128', 'shake_256')


def hash_file(file_name, method):
    assert hashlib
    m = eval(f'hashlib.{method}')()
    with open(file_name, 'rb') as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            m.update(data)
    return m.hexdigest()


def hash_str(content, method):
    assert hashlib
    m = eval(f'hashlib.{method}')()
    m.update(content.encode(encoding='utf-8'))
    return m.hexdigest()


def md5_file(path):
    with open(path, 'rb') as fp:
        data = fp.read()
    return hashlib.md5(data).hexdigest()


def md5_str(s):
    md5hash = hashlib.md5(s)
    return md5hash.hexdigest()
