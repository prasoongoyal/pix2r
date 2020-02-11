import torch
from torch.utils.data import Dataset
import string
translator = str.maketrans('', '', string.punctuation)
import random
import glob
from PIL import Image
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import pickle

DATA_DIR = '../../data/'
FRAMES_DIR = '../../data/frames/'
ENVS_DIR = '../../data/envs/'

class PadBatch:
  def __init__(self):
    pass

  def __call__(self, batch):
    traj_right_batch, traj_left_batch, traj_center_batch, \
    lang_batch, lang_enc_batch, traj_len_batch, lang_len_batch, labels_batch, obj_batch, \
        env_batch, weight_batch = zip(*batch)
    traj_right_batch = pad_sequence(traj_right_batch, batch_first=True)
    traj_left_batch = pad_sequence(traj_left_batch, batch_first=True)
    traj_center_batch = pad_sequence(traj_center_batch, batch_first=True)
    lang_enc_batch = pad_sequence(lang_enc_batch, batch_first=True)
    lang_enc_batch = torch.from_numpy(np.array(lang_enc_batch))
    traj_len_batch = torch.Tensor(traj_len_batch)
    lang_len_batch = torch.Tensor(lang_len_batch)
    weight_batch = torch.Tensor(weight_batch)
    labels_batch = torch.Tensor(labels_batch)
    return traj_right_batch, traj_left_batch, traj_center_batch, \
        lang_batch, lang_enc_batch, traj_len_batch, lang_len_batch, \
        labels_batch, obj_batch, env_batch, weight_batch

class Data(Dataset):
  def __init__(self, mode, sampling, prefix, repeat=1):
    self.sampling = sampling
    self.prefix = prefix
    self.vocab = pickle.load(open('{}/vocab_train.pkl'.format(DATA_DIR), 'rb'))
    self.descriptions = self.load_descriptions(mode)
    if mode == 'train':
      self.video_ids = list(range(80))
      self.video_ids.remove(50)
    elif mode == 'valid':
      self.video_ids = list(range(80, 100))
      self.video_ids.remove(92)
      self.video_ids.remove(95)
    else:
      raise NotImplementedError('Invalid mode!')
    self.N_OBJ = 13
    self.N_ENV = len(self.video_ids)
    self.thresh = 0.
    self.len_frac = 0.
    self.repeat = repeat

  def update_len_frac(self):
    self.len_frac = max(0, self.len_frac - 0.1)

  def __len__(self):
    return 2 * self.N_OBJ * self.N_ENV * self.repeat

  def set_thresh(self, thresh):
    self.thresh = thresh

  def encode_description(self, descr):
    result = []
    for w in descr.split():
        try:
            t = self.vocab.index(w)
        except ValueError:
            t = self.vocab.index('<unk>')
        result.append(t)
    return torch.Tensor(result)

  def load_descriptions(self, mode):
    descriptions = pickle.load(open('{}/{}_descr.pkl'.format(DATA_DIR, mode), 'rb'))
    result = {}
    for i in descriptions.keys():
        descr_list = descriptions[i]
        result[i] = [(d, self.encode_description(d)) for d in descr_list]
    return result

  def load_env_objects(self, obj, env):
    result = []
    with open('{}/obj{}-env{}.txt'.format(ENVS_DIR, obj, env)) as f:
      for line in f.readlines():
        line = line.replace('(', '').replace(',', '').replace(')', '')
        parts = line.split()
        x = eval(parts[0])
        y = eval(parts[1])
        obj = eval(parts[2])
        result.append(obj)
    return result

  def __getitem__(self, index):
    if index >= len(self) // 2:
      label = 0
      index -= len(self) // 2
    else:
      label = 1

    obj = index // (self.repeat * self.N_ENV)
    env = self.video_ids[index % self.N_ENV]
    env_objects = self.load_env_objects(obj, env)

    frames_right = torch.from_numpy(torch.load(open('{}/obj{}-env{}-right-50x50.pt'.format(FRAMES_DIR, obj, env), 'rb')))
    frames_left = torch.from_numpy(torch.load(open('{}/obj{}-env{}-left-50x50.pt'.format(FRAMES_DIR, obj, env), 'rb')))
    frames_center = torch.from_numpy(torch.load(open('{}/obj{}-env{}-center-50x50.pt'.format(FRAMES_DIR, obj, env), 'rb')))

    lower = max(1, self.len_frac * (len(frames_right)+1) - 1)
    t = np.random.randint(lower, (len(frames_right)+1))
    weight = (t / len(frames_right))**1
    frames_right = frames_right[:t]
    frames_left = frames_left[:t]
    frames_center = frames_center[:t]

    while True:
      selected = np.random.random(len(frames_right)) > 0.9
      if np.sum(selected) > 0:
        break
    frames_right = frames_right[selected]
    frames_left = frames_left[selected]
    frames_center = frames_center[selected]
      
    if label == 1:
      t = np.random.randint(0, len(self.descriptions[obj]))
      descr, descr_enc = self.descriptions[obj][t]
    else:
      tt = np.random.random()
      # alternate lang
      alt_obj = env_objects[1:]
      # alt_obj = list(filter(lambda xyo: abs(env_objects[0][0] - xyo[0]) >= 2, env_objects))
      # alt_obj = list(map(lambda xyo: xyo[-1], alt_obj))
      if len(alt_obj) == 0:
        alt_obj = list(range(0,obj)) + list(range(obj+1,self.N_OBJ))
      obj_ = np.random.choice(alt_obj)
      t = np.random.randint(0, len(self.descriptions[obj_])-1)
      descr, descr_enc = self.descriptions[obj_][t]
    label = 2*label - 1
    return frames_right, frames_left, frames_center, descr, descr_enc, len(frames_right), len(descr_enc), label, obj, env, weight

