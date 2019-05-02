import urllib.parse
from sklearn import tree
import io
import string
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


def parse_file(file_in, file_out):
    fin = open(file_in)
    fout = io.open(file_out, "w", encoding="utf-8")
    lines = fin.readlines()

    for i in range(len(lines)):
        line = lines[i].strip()
        res = ""
        if line.startswith("GET"):
            # Decode the URL from the GET requests
            res += "GET" + line.split(" ")[1]
            val = decode_url(res)
            fout.writelines(val + '\n')
        elif line.startswith("POST") or line.startswith("PUT"):
            # Decode the URL from the POST and PUT requests
            url = line.split(' ')[0] + line.split(' ')[1]
            val = decode_url(url)
            fout.writelines(val + '\n')
        elif lines[i - 2].startswith("Content-Length"):
            # Decode the url at the bottom of the POST request
            val = decode_url(line)
            fout.writelines(val + '\n')
        elif line == "" and (lines[i - 1].startswith("Content-Length") or lines[i - 1].strip() == ""):
            # To remove the space between the POST/PUT request and the ID at the bottom
            # Also, to remove the double "\n" between GET requests
            continue
        else:
            # Write all other lines without decoding
            fout.writelines(line + '\n')

    print("finished parse")
    fout.close()
    fin.close()


def decode_url(line):
    # Gets rid of escape characters in the URL
    # requests will be in lower case
    line = urllib.parse.unquote(line).replace('\n', '').lower()
    return line

def to_string(file):
    list_requests = []
    with open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if (len(d) > 0):
            result.append(d)
        else:
            list_requests.append("\n".join(result))
            result = []
            continue
    return list_requests

def encode_data(x, maxlen, vocab):
    # Iterate over the loaded data and create a matrix of size (len(x), maxlen)
    # Each character is encoded into a one-hot array later at the lambda layer.
    # Chars not in the vocab are encoded as -1, into an all zero vector.
    # Account for unicode ---------------- some characters are in spanish

    input_data = np.zeros((len(x), maxlen), dtype=np.int)
    for dix, sent in enumerate(x):
        counter = 0
        for c in sent:
            c=c.lower()
            if counter >= maxlen:
                pass
            else:
                ix = vocab.get(c, -1)  # get index from vocab dictionary, if not in vocab, return -1
                input_data[dix, counter] = ix
                counter += 1
    return input_data


def create_vocab_set():
    # This alphabet is 69 chars vs. 70 reported in the paper since they include two
    # '-' characters. See https://github.com/zhangxiangxiao/Crepe#issues.

    alphabet = set(list(string.ascii_lowercase) + list(string.digits) +
                   list(string.punctuation) + ['\n'])
    vocab_size = len(alphabet)
    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, alphabet

def load_data(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=21)
    return (x_train, y_train), (x_test, y_test)