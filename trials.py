import sys
import torch
import torch.nn as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_sequences(filename, sep="\t"):
    """
    Outputs all the sentences in the tsv format like:

    [['Burgdorf', 'PER'], ['had', 'O'],
    ['brought', 'O'], ['a', 'O'],
    ['capsule', 'O'], ['of', 'O'],
    ['cyanide', 'O'], ['for', 'O'],
    ['the', 'O'], ['occasion', 'O'],
    ['.', 'O']]

    """
    sequences = []
    with open(filename, encoding='utf-8') as fp:
        seq = []
        for line in fp:
            line = line.strip()
            if line:
                line = line.split(sep)
                if line[0][0] != '#':
                    seq.append(line[1:])
            else:
                sequences.append(seq)
                seq = []
        if seq:
            sequences.append(seq)
    return sequences


# Input: "Output of load sequences" -> Output: "Only the sentences to be used in padding operation (Tokenized) "
def get_sentences(sequences):
    X = [[w[0] for w in s] for s in sequences]
    new_X = []
    for seq in X:
        new_X.append(seq)
    return new_X


def get_tags(sequences):
    X = [[w[1] for w in s] for s in sequences]
    new_Y = []
    for seq in X:
        new_Y.append(seq)
    return new_Y


# Gets the inner list of list of list structure and turns them into torch tensors
def list2tensor(X):
    for i in range(len(X)):
        X[i] = torch.tensor(X[i])
    return X


def prepare_data(path):
    sequences = load_sequences(path)
    sentences = get_sentences(sequences)
    tags = get_tags(sequences)

    words = [list[0] for lol in sequences for list in lol]
    word_vocab = np.unique(words)

    word2ind = {w: i + 2 for i, w in enumerate(word_vocab)}
    word2ind["PAD"] = 0
    word2ind["UNK"] = 1

    ind2word = {i: w for w, i in word2ind.items()}

    tags2 = [list[1] for lol in sequences for list in lol]
    tags_vocab = np.unique(tags2)

    tags2index = {t: i + 1 for i, t in enumerate(tags_vocab)}
    tags2index["PAD"] = 0
    ind2tag = {i: w for w, i in tags2index.items()}
    X = [[word2ind[w] for w in s] for s in sentences]
    X = list2tensor(X)
    X = pad_sequence(sequences=X, batch_first=True, padding_value=word2ind["PAD"])
    # Getting labels to pad
    Y = [[tags2index[w] for w in s] for s in tags]
    Y = list2tensor(Y)
    Y = pad_sequence(sequences=Y, batch_first=True, padding_value=tags2index["PAD"])

    return X, Y, word2ind, ind2tag, tags2index, word_vocab, ind2word


x_train, y_train, w2i, i2t, t2i, _, i2w = prepare_data('/home/beno/nlp2020-hw1/data/train.tsv')
x_test, y_test, w2i_test, i2t_test, t2i_test, _, i2w_test = prepare_data('/home/beno/nlp2020-hw1/data/test.tsv')
x_dev, y_dev, w2i_dev, i2t_dev, t2i_dev, _, i2w_dev = prepare_data('/home/beno/nlp2020-hw1/data/dev.tsv')


# log = open("BILSTM+CRF.log", "w")
#####################################################################
# Helper functions to make the code more readable.


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


#####################################################################
# Create model


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, word_embedding_dim,
                  hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding(vocab_size, word_embedding_dim)



        self.lstm = nn.LSTM(word_embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim //2).to(device),
                torch.randn(2, 1, self.hidden_dim //2).to(device))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function

        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.).to(device=device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1).to(device=device)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        # self.hidden = self.init_hidden()
        # combined_word_embeds = []
        # for item in sentence:
        #     tmp_embeds = self.word_embeds(item).view(len(item), -1)
        #     combined_word_embeds.append(torch.sum(tmp_embeds, 0))
        # word_embeds = torch.cat(combined_word_embeds)
        # embeds = word_embeds.view(len(combined_word_embeds), 1, -1)
        # lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # lstm_out = lstm_out.view(len(combined_word_embeds), self.hidden_dim * 2)
        # lstm_feats = self.hidden2tag(lstm_out)
        # return lstm_feats

        sentence = sentence.to(device=device)

        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(device)

        tags = tags.to(device)

        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]).to(device=device), tags])

        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):

        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


def evaluate(instances, model):
    #    log.write("evaluation start...\n")
    print
    "evaluation start..."
    right = 0.
    total = 0.
    idx = 0
    for sentences,tags in instances:
        idx += 1
        if idx % 100 == 0:
            #            log.write(str(idx)+"\n")
            print
            idx

        a = []
        b= []
        a.append(sentences)
        b.append(tags)

        packed_instance = []
        packed_instance.append([autograd.Variable(torch.LongTensor(x)) for x in a])
        packed_instance.append([autograd.Variable(torch.LongTensor(x)) for x in b])


        print(packed_instance)

        _, tag_seq = model(packed_instance)


        assert len(tag_seq) == len(tags)

        for i in range(len(tag_seq)):
            if tag_seq[i] == tags[i]:
                right += 1
        total += len(tag_seq)
    return right / total


#####################################################################
# Run training
trn_filename = "/home/beno/nlp2020-hw1/data/train.tsv"
dev_filename = "/home/beno/nlp2020-hw1/data/dev.tsv"
tst_filename = "/home/beno/nlp2020-hw1/data/test.tsv"

START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNK = "<UNK>"
WORD_EMBEDDING_DIM = 64
HIDDEN_DIM = 100

#########################################################
# Load training data

tag_to_ix = {}

word_to_ix = w2i
tag_to_ix = t2i
##########################################################
###########################################################
# Load dev data
word_to_ix_dev = w2i_dev
tag_to_ix_dev = t2i_dev
###########################################################
# Load tst data
word_to_ix_test = w2i_test
tag_to_ix_test = w2i_test

tag_to_ix[START_TAG] = len(tag_to_ix)
tag_to_ix[STOP_TAG] = len(tag_to_ix)

# log.write("word dict size: " + str(len(word_to_ix))+"\n")
# log.write("lemma dict size: " + str(len(lemma_to_ix))+"\n")
# log.write("tag dict size: "+ str(len(tag_to_ix))+"\n")



model = BiLSTM_CRF(len(word_to_ix) +1, tag_to_ix,
                   WORD_EMBEDDING_DIM, HIDDEN_DIM).to(device=device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-4)

###########################################################

# Check predictions before training
# log.write("DEV accuracy= " + str(evaluate(dev_instances, model))+"\n")
# print("DEV accuracy= " + str(evaluate(dev_instances, model))
# print("DEV accuracy= " + str(evaluate(dev_instances, model))
# log.write("TST accuracy= " + str(evaluate(tst_instances, model))+"\n")
# log.flush()

total_loss = 0

timestep = 0

trn_see = 1000
eval_see = 10
# print(model.pretrain_embeds(autograd.Variable(torch.LongTensor([7]))))
# print(model.word_embeds(autograd.Variable(torch.LongTensor([1]))))
# Training loop
for epoch in range(1):
    for sentences, tags in tqdm(zip(x_train[:1000],y_train[:1000]),total=len(x_train[:1000])):

        model.zero_grad()

        neg_log_likelihood = model.neg_log_likelihood(sentences, tags)
        total_loss += neg_log_likelihood
        neg_log_likelihood.backward()
        optimizer.step()

        # print(model.pretrain_embeds(autograd.Variable(torch.LongTensor([7]))))
        # print(model.word_embeds(autograd.Variable(torch.LongTensor([1]))))
        # exit(1)
        if timestep % trn_see == 0:
            print("epoch: " + str(timestep * 1.0 / len(x_train)) + " loss: " + str(to_scalar(total_loss) / trn_see))
            total_loss = 0
            # print(model.word_embeds(autograd.Variable(torch.LongTensor([1]))))
            # print(model.pretrain_embeds(autograd.Variable(torch.LongTensor([7]))))
            # log.flush()

        # if timestep % (trn_see * eval_see) == 0:
        #     # log.write("DEV accuracy= " + str(evaluate(dev_instances, model))+"\n")
        #     print
        #     "DEV accuracy= " + str(evaluate(zip(x_dev,y_dev), model))
        #     #        log.write("TST accuracy= " + str(evaluate(tst_instances, model))+"\n")
        #     torch.save(model.state_dict(), "/home/beno/nlp2020-hw1/data/model." + str(timestep / (trn_see * eval_see)))
            # log.flush()
    # log.close()
