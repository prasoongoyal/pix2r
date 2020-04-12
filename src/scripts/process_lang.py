import pickle
import string
translator = str.maketrans('', '', string.punctuation)

def split_descr():
  descr = {}
  # read all descriptions
  with open('all_descr.txt') as f:
    for line in f.readlines():
      line = line.strip()
      line = line.translate(translator).lower()
      parts = line.split('\t')
      try:
        descr_list = descr[eval(parts[0])]
      except:
        descr_list = []
      # print(parts[0], eval(parts[0]))
      descr[eval(parts[0])] = descr_list + [parts[-1]]

  # split
  train = {}
  valid = {}
  test = {}
  for i in descr.keys():
    descr_list = descr[i]
    train[i] = descr_list[:-8]
    valid[i] = descr_list[-8:-3]
    test[i] = descr_list[-3:]

  # print(train)
  # print(valid)
  # print(test)

  return train, valid, test

def create_vocab(descr):
  vocab = {}
  for i in descr.keys():
    descr_list = descr[i]
    for j in range(len(descr_list)):
        words = descr_list[j].split()
        for w in words:
          try:
            freq = vocab[w]
          except:
            freq = 0
          vocab[w] = freq+1
  return vocab

def main():
  train, valid, test = split_descr()
  vocab = create_vocab(train)
  vocab = ['<unk>'] + list(vocab.keys())
  print(len(vocab))
  pickle.dump(train, open('train_descr.pkl', 'wb'))
  pickle.dump(valid, open('valid_descr.pkl', 'wb'))
  pickle.dump(test, open('test_descr.pkl', 'wb'))
  pickle.dump(vocab, open('vocab_train.pkl', 'wb'))
  return

if __name__ == '__main__':
  main()
