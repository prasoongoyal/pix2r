from data_alfred import Data, PadBatch
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.stats import spearmanr


VOCAB_SIZE = 447

def lstm_helper(sequences, lengths, lstm):
    if len(sequences) == 1:
        output, (hidden, _) = lstm(sequences)
        return output, hidden[-1]

    ordered_len, ordered_idx = lengths.sort(0, descending=True)
    ordered_sequences = sequences[ordered_idx]
    # remove zero lengths
    try:
        nonzero = list(ordered_len).index(0)
    except ValueError:
        nonzero = len(ordered_len)

    sequences_packed = pack_padded_sequence(
        ordered_sequences[:nonzero], ordered_len[:nonzero],
        batch_first=True)
    output_nonzero, (hidden_nonzero, _) = lstm(sequences_packed)
    output_nonzero = pad_packed_sequence(output_nonzero, batch_first=True)[0]
    max_len = sequences.shape[1]
    max_len_true = output_nonzero.shape[1]
    output = torch.zeros(len(sequences), max_len, output_nonzero.shape[-1])
    output_final = torch.zeros(
        len(sequences), max_len, output_nonzero.shape[-1])
    output[:nonzero, :max_len_true, :] = output_nonzero
    hidden = torch.zeros(len(sequences), hidden_nonzero.shape[-1])
    hidden_final = torch.zeros(len(sequences), hidden_nonzero.shape[-1])
    hidden[:nonzero, :] = hidden_nonzero[-1]
    output_final[ordered_idx] = output
    hidden_final[ordered_idx] = hidden
    return output_final.cuda(), hidden_final.cuda()

class ImgEnc(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = nn.Sequential(
            nn.Conv2d(3, args.n_channels, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(args.n_channels, args.n_channels, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(args.n_channels, args.n_channels, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(4*4*args.n_channels, args.img_enc_size),
            nn.Linear(args.img_enc_size, args.img_enc_size),
        )

    def forward(self, x):
        return self.encoder(x)

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.img_enc = ImgEnc(args)
        traj_enc_size = args.img_enc_size

        '''
        if args.view == 'all':
            self.img_enc_r = ImgEnc(args)
            self.img_enc_l = ImgEnc(args)
            self.img_enc_c = ImgEnc(args)
            traj_enc_size = 3 * args.img_enc_size
        elif args.view == 'right':
            self.img_enc_r = ImgEnc(args)
            traj_enc_size = args.img_enc_size
        elif args.view == 'center':
            self.img_enc_c = ImgEnc(args)
            traj_enc_size = args.img_enc_size
        elif args.view == 'left':
            self.img_enc_l = ImgEnc(args)
            traj_enc_size = args.img_enc_size
        else:
            raise NotImplementedError('Invalid view!')
        '''

        if args.loss == 'rgr':
            output_size = 1
        elif args.loss == 'cls':
            output_size = 2
        else:
            raise NotImplementedError('Invalid loss type!')

        # trajectory encoder
        if not args.meanpool_traj:
            self.traj_encoder = nn.LSTM(
                traj_enc_size, 
                traj_enc_size, 
                batch_first=True, 
                num_layers=args.num_layers)

        # language encoder
        self.embedding = nn.Embedding(VOCAB_SIZE, args.lang_enc_size)
        if not args.meanpool_lang:
            self.lang_encoder = nn.LSTM(
                args.lang_enc_size, 
                args.lang_enc_size, 
                batch_first=True, 
                num_layers=args.num_layers)

        # linear layers
        self.linear1 = nn.Linear(traj_enc_size + args.lang_enc_size, args.classifier_size)
        self.linear2 = nn.Linear(args.classifier_size, output_size)


    def forward(self, traj, lang, traj_len, lang_len):
        # import pdb
        # pdb.set_trace()
        traj_enc = self.img_enc(traj.view(-1, *traj.shape[-3:]))
        traj_enc = traj_enc.view(*traj.shape[:2], -1)

        '''
        if self.args.view == 'all':
            traj_r_enc = self.img_enc_r(traj_r.view(-1, *traj_r.shape[-3:]))
            traj_r_enc = traj_r_enc.view(*traj_r.shape[:2], -1)
            traj_l_enc = self.img_enc_l(traj_l.view(-1, *traj_l.shape[-3:]))
            traj_l_enc = traj_l_enc.view(*traj_l.shape[:2], -1)
            traj_c_enc = self.img_enc_c(traj_c.view(-1, *traj_c.shape[-3:]))
            traj_c_enc = traj_c_enc.view(*traj_c.shape[:2], -1)
            traj_enc = torch.cat([traj_r_enc, traj_l_enc, traj_c_enc], dim=-1)

        elif self.args.view == 'right':
            traj_enc = self.img_enc_r(traj_r.view(-1, *traj_r.shape[-3:]))
            traj_enc = traj_enc.view(*traj_r.shape[:2], -1)

        elif self.args.view == 'center':
            traj_enc = self.img_enc_c(traj_c.view(-1, *traj_c.shape[-3:]))
            traj_enc = traj_enc.view(*traj_c.shape[:2], -1)

        elif self.args.view == 'left':
            traj_enc = self.img_enc_l(traj_l.view(-1, *traj_l.shape[-3:]))
            traj_enc = traj_enc.view(*traj_l.shape[:2], -1)

        else:
            raise NotImplementedError('Invalid view!')
        '''

        if self.args.meanpool_traj:
            traj_len = traj_len.long()
            traj_enc = torch.stack(
                [torch.mean(traj_enc[i][:traj_len[i]], dim=0) for i in range(len(traj_enc))])
        else:
            _, traj_enc = lstm_helper(traj_enc, traj_len, self.traj_encoder)

        lang_emb = self.embedding(lang)
        if self.args.meanpool_lang:
            lang_len = lang_len.long()
            lang_enc = torch.stack(
                [torch.mean(lang_emb[i][:lang_len[i]], dim=0) for i in range(len(lang_emb))])
        else:
            _, lang_enc = lstm_helper(lang_emb, lang_len, self.lang_encoder)

        traj_lang = torch.cat([traj_enc, lang_enc], dim=-1)
        pred = F.relu(self.linear1(traj_lang))
        pred = self.linear2(pred)
        # print(pred)
        return pred, lang_emb

    def forward_enc(self, traj_r, traj_l, traj_c, lang, traj_len, lang_len):
        traj_enc = torch.cat([traj_r, traj_l, traj_c], dim=-1)
        if self.args.meanpool_traj:
            traj_len = traj_len.long()
            traj_enc = torch.stack(
                [torch.mean(traj_enc[i][:traj_len[i]], dim=0) for i in range(len(traj_enc))])
        else:
            _, traj_enc = lstm_helper(traj_enc, traj_len, self.traj_encoder)

        lang_emb = self.embedding(lang)
        if self.args.meanpool_lang:
            lang_len = lang_len.long()
            lang_enc = torch.stack(
                [torch.mean(lang_emb[i][:lang_len[i]], dim=0) for i in range(len(lang_emb))])
        else:
            _, lang_enc = lstm_helper(lang_emb, lang_len, self.lang_encoder)

        traj_lang = torch.cat([traj_enc, lang_enc], dim=-1)
        pred = F.relu(self.linear1(traj_lang))
        pred = self.linear2(pred)
        return pred, lang_emb

class Predict:
    def __init__(self, model_file, lr, n_updates):
        ckpt = torch.load(model_file)
        self.args = ckpt['args']
        self.model = Model(self.args).cuda()
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=0.)
        self.n_updates = n_updates
        self.model.eval()

    def encode_frames(self, frame_r, frame_l, frame_c):
        with torch.no_grad():
            frame_r = torch.from_numpy(frame_r).float().cuda()
            frame_r = torch.transpose(frame_r, 1, 2)
            frame_r = torch.transpose(frame_r, 0, 1)
            frame_r = self.model.img_enc_r(torch.unsqueeze(frame_r, 0) / 255.)
            frame_l = torch.from_numpy(frame_l).float().cuda()
            frame_l = torch.transpose(frame_l, 1, 2)
            frame_l = torch.transpose(frame_l, 0, 1)
            frame_l = self.model.img_enc_l(torch.unsqueeze(frame_l, 0) / 255.)
            frame_c = torch.from_numpy(frame_c).float().cuda()
            frame_c = torch.transpose(frame_c, 1, 2)
            frame_c = torch.transpose(frame_c, 0, 1)
            frame_c = self.model.img_enc_c(torch.unsqueeze(frame_c, 0) / 255.)
        return torch.squeeze(frame_r).cpu(), torch.squeeze(frame_l).cpu(), torch.squeeze(frame_c).cpu()

    def predict(self, traj_r, traj_l, traj_c, lang):
        with torch.no_grad():
            traj_r_sampled = traj_r[::-1][::10][::-1]
            traj_r_sampled = np.array(traj_r_sampled)
            traj_r_sampled = torch.from_numpy(traj_r_sampled)
            traj_r_sampled = traj_r_sampled.cuda().float()
            traj_r_sampled = torch.transpose(traj_r_sampled, 2, 3)
            traj_r_sampled = torch.transpose(traj_r_sampled, 1, 2)
            traj_l_sampled = traj_l[::-1][::10][::-1]
            traj_l_sampled = np.array(traj_l_sampled)
            traj_l_sampled = torch.from_numpy(traj_l_sampled)
            traj_l_sampled = traj_l_sampled.cuda().float()
            traj_l_sampled = torch.transpose(traj_l_sampled, 2, 3)
            traj_l_sampled = torch.transpose(traj_l_sampled, 1, 2)
            traj_c_sampled = traj_c[::-1][::10][::-1]
            traj_c_sampled = np.array(traj_c_sampled)
            traj_c_sampled = torch.from_numpy(traj_c_sampled)
            traj_c_sampled = traj_c_sampled.cuda().float()
            traj_c_sampled = torch.transpose(traj_c_sampled, 2, 3)
            traj_c_sampled = torch.transpose(traj_c_sampled, 1, 2)
            lang = lang.cuda().long()
            traj_len = torch.Tensor([len(traj_r_sampled)])
            lang_len = torch.Tensor([len(lang)])
            prob, _ = self.model(
                torch.unsqueeze(traj_r_sampled, 0) / 255., 
                torch.unsqueeze(traj_l_sampled, 0) / 255.,
                torch.unsqueeze(traj_c_sampled, 0) / 255., 
                torch.unsqueeze(lang, 0), traj_len, lang_len)
        if self.args.loss == 'cls':
            prob = torch.softmax(prob, dim=-1).data.cpu().numpy()[0]
            return prob[1] - prob[0]
        elif self.args.loss == 'rgr':
            tt = torch.tanh(prob)
            # prob = tt.item()
            # import pdb
            # pdb.set_trace()
            # tt = tt.data
            # tt = tt.cpu()
            # tt = tt.numpy()
            prob = tt[0][0]
            # prob = torch.tanh(prob).data.cpu().numpy()[0][0]
            return prob
        else:
            raise NotImplementedError('Invalid loss type!')
        # return prob

    def predict_enc(self, traj_r, traj_l, traj_c, lang):
        with torch.no_grad():
            traj_r_sampled = torch.stack(traj_r[::-1][::10][::-1]).float().cuda()
            traj_l_sampled = torch.stack(traj_l[::-1][::10][::-1]).float().cuda()
            traj_c_sampled = torch.stack(traj_c[::-1][::10][::-1]).float().cuda()
            lang = lang.cuda().long()
            traj_len = torch.Tensor([len(traj_r_sampled)])
            lang_len = torch.Tensor([len(lang)])
            prob, _ = self.model.forward_enc(
                torch.unsqueeze(traj_r_sampled, 0), 
                torch.unsqueeze(traj_l_sampled, 0),
                torch.unsqueeze(traj_c_sampled, 0), 
                torch.unsqueeze(lang, 0), traj_len, lang_len)
        if self.args.loss == 'cls':
            prob = torch.softmax(prob, dim=-1).data.cpu().numpy()[0]
            return prob[1] - prob[0]
        elif self.args.loss == 'rgr':
            prob = torch.tanh(prob).data.cpu().numpy()[0][0]
            return prob
        else:
            raise NotImplementedError('Invalid loss type!')
        # return prob

    def update(self, traj_r, traj_l, traj_c, lang, label):
        self.model.train()
        traj_len = min(150, len(traj_r))
        traj_r = torch.from_numpy(np.array(traj_r[:traj_len]))
        traj_r = torch.transpose(traj_r, 2, 3)
        traj_r = torch.transpose(traj_r, 1, 2)
        traj_l = torch.from_numpy(np.array(traj_l[:traj_len]))
        traj_l = torch.transpose(traj_l, 2, 3)
        traj_l = torch.transpose(traj_l, 1, 2)
        traj_c = torch.from_numpy(np.array(traj_c[:traj_len]))
        traj_c = torch.transpose(traj_c, 2, 3)
        traj_c = torch.transpose(traj_c, 1, 2)
        lang = lang.cuda().long()
        lang_len = torch.Tensor([len(lang)])
        label = torch.Tensor([2*label - 1]).cuda()
        for _ in range(self.n_updates):
            while True:
                selected = np.random.random(traj_len) > 0.9
                if np.sum(selected) > 0:
                    break
            traj_r_ = traj_r[selected].cuda().float()
            traj_l_ = traj_l[selected].cuda().float()
            traj_c_ = traj_c[selected].cuda().float()
            traj_len_ = torch.Tensor([len(traj_r_)])
            self.optimizer.zero_grad()
            prob = self.model(
                torch.unsqueeze(traj_r_, 0) / 255.,
                torch.unsqueeze(traj_l_, 0) / 255., 
                torch.unsqueeze(traj_c_, 0) / 255.,
                torch.unsqueeze(lang, 0),
                traj_len_, lang_len)[:, 0]
            loss = torch.nn.MSELoss()(prob, label)
            loss.backward()
            self.optimizer.step()

class Train:
    def __init__(self, args, train_data_loader, valid_data_loader):
        self.args = args
        self.model = Model(args).cuda()
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.args.lr)

    def run_batch(self, traj, lang, traj_len, 
        lang_len, classes, weights, is_train):
        if is_train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        traj = traj.cuda().float()
        # traj_l = traj_l.cuda().float()
        # traj_c = traj_c.cuda().float()
        lang = lang.cuda().long()
        classes = torch.Tensor(classes).cuda().long()
        weights = weights.cuda().float()
        pred, _ = self.model(traj, lang, traj_len, lang_len)
        if self.args.loss == 'rgr':
            pred = torch.tanh(pred[:, 0])
            labels = weights*classes
            loss = torch.nn.MSELoss()(pred, labels)
            # loss = torch.nn.L1Loss()(pred, labels)
        elif self.args.loss == 'cls':
            labels = (classes + 1) / 2
            loss = torch.nn.CrossEntropyLoss()(pred, labels)
            pred = torch.argmax(pred, dim=-1)
        else:
            raise NotImplementedError('Invalid loss type!')

        if is_train:
            loss.backward()
            self.optimizer.step()

        return pred, labels, loss.item()

    def run_epoch(self, data_loader, is_train):
        pred_all = []
        labels_all = []
        loss_all = []

        p = np.zeros(4)
        l = np.zeros(4)
        # traj_len_all = []
        for index, frames, descr, frames_len, descr_len, weights, labels in data_loader:
            pred, labels, loss = self.run_batch(frames, descr, frames_len, descr_len, labels, weights, is_train)
        # for _, frames_r, frames_l, frames_c, descr, descr_enc, \
        #     traj_len, descr_len, classes, _, _, weights in data_loader:
        #     pred, labels, loss = self.run_batch(
        #         frames_r, frames_l, frames_c, descr_enc, traj_len, 
        #         descr_len, classes, weights, is_train)
            # print(index, descr, frames_len, descr_len, weights, labels)
            # import pdb
            # pdb.set_trace()
            pred_all += pred.tolist()
            labels_all += labels.tolist()
            loss_all.append(loss)
            # for i, pi, li in zip(index.tolist(), pred.tolist(), labels.tolist()):
            #     p[i] = pi
            #     l[i] = li
            # traj_len_all += traj_len.tolist()
        # if is_train:
        #     print(p)
        #     import pdb
        #     pdb.set_trace()
            # for i, (pi, li) in enumerate(zip(p, l)):
            #     print(pi, li)
        if self.args.loss == 'rgr':
            # print(pred_all, labels_all)
            score, _ = spearmanr(pred_all, labels_all)
        elif self.args.loss == 'cls':
            correct = [1 if p==l else 0 for (p, l) in zip(pred_all, labels_all)]
            score = sum(correct) / len(correct)
        else:
            raise NotImplementedError('Invalid loss type!')
        return np.mean(loss_all), score

    def train_model(self):
        best_val_acc = 0.
        epoch = 1
        while True:
            valid_loss, valid_acc = self.run_epoch(
                self.valid_data_loader, is_train=False)
            train_loss, train_acc = self.run_epoch(
                self.train_data_loader, is_train=True)
            print('Epoch: {}\tTL: {:.8f}\tTA: {:.2f}\tVL: {:.2f}\tVA: {:.2f}'.format(
                epoch, train_loss, 100. * train_acc, valid_loss, 100. * valid_acc))
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                if self.args.save_path:
                    state = {
                        'args': self.args, 
                        'epoch': epoch, 
                        'best_val_acc': best_val_acc, 
                        'state_dict': self.model.state_dict(), 
                        # 'optimizer': self.optimizer.state_dict()
                    }                
                    torch.save(state, self.args.save_path)
            if epoch == self.args.max_epochs:
                break
            epoch += 1

def main(args):
    if args.debug:
        repeat = 1
    else:
        repeat = 10
    train_data = Data(args, mode='train', repeat=repeat)
    valid_data = Data(args, mode='valid_seen', repeat=repeat)
    print(len(train_data))
    print(len(valid_data))
    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=PadBatch(),
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)),
        num_workers=args.num_workers)
    valid_data_loader = DataLoader(
        dataset=valid_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=PadBatch(),
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)),
        num_workers=args.num_workers)
    Train(args, train_data_loader, valid_data_loader).train_model()

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--loss', default='rgr', help='rgr | cls')
    parser.add_argument('--view', default='all', help='all | left | center | right')
    parser.add_argument('--meanpool-lang', action='store_true')
    parser.add_argument('--meanpool-traj', action='store_true')
    parser.add_argument('--last-frame', action='store_true')
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--max-epochs', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--logdir', default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hyperparam_values = [64, 96, 128, 192, 256, 384, 512]
    args.n_channels = hyperparam_values[np.random.randint(len(hyperparam_values))]
    args.img_enc_size = hyperparam_values[np.random.randint(len(hyperparam_values))]
    args.lang_enc_size = hyperparam_values[np.random.randint(len(hyperparam_values))]
    args.classifier_size = hyperparam_values[np.random.randint(len(hyperparam_values))]
    print(args)
    main(args)
