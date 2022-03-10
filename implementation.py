import numpy as np
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence
import torch
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)

# from model import Model

#
#
# def build_model(device: str) -> Model:
#     # STUDENT: return StudentModel()
#     # STUDENT: your model MUST be loaded on the device "device" indicates
#     return RandomBaseline()
#
#
# class RandomBaseline(Model):
#
#     options = [
#         ('LOC', 98412),
#         ('O', 2512990),
#         ('ORG', 71633),
#         ('PER', 115758)
#     ]
#
#     def __init__(self):
#
#         self._options = [option[0] for option in self.options]
#         self._weights = np.array([option[1] for option in self.options])
#         self._weights = self._weights / self._weights.sum()
#
#     def predict(self, tokens: List[List[str]]) -> List[List[str]]:
#         return [[str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x] for x in tokens]
#
#
# class StudentModel(Model):
#
#     # STUDENT: construct here your model
#     # this class should be loading your weights and vocabulary
#
#     def predict(self, tokens: List[List[str]]) -> List[List[str]]:
#         # STUDENT: implement here your predict function
#         # remember to respect the same order of tokens!
#         pass


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast))).to(device=device)


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))





        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, MINIBATCH_SIZE, self.hidden_dim // 2).to(device),
                torch.randn(2, MINIBATCH_SIZE, self.hidden_dim // 2).to(device))

    def _forward_alg(self, feats):
        feats.to(device)
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(device=device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
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
        self.hidden = self.init_hidden()

        combined_word_embeds = []
        for item in sentence:
            tmp_embeds = self.word_embeds(item).view(len(item), -1)
            combined_word_embeds.append(torch.sum(tmp_embeds, 0))
        word_embeds = torch.cat(combined_word_embeds)

        embeds = word_embeds.view(len(sentence),1,-1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)  # hidden_dim -> 4
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]).to(device))
        comb_tags = []
        for item in tags:
            tmp_tags = tags.view(len(item), -1)
            comb_tags.append(torch.sum(tmp_tags, 0))

        tagsN = torch.cat(comb_tags)

        tagsN = tagsN.to(device)
        tagsN = tagsN.long()
        tagsN = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(device), tagsN])

        print(tagsN.shape)


        comb_feats = []
        for item in feats:
            tmp_feats = feats.view(len(item), -1)
            comb_feats.append(torch.sum(tmp_feats, 0))
        featsN = torch.cat(comb_feats)
        featsN = featsN.to(device)

        print(featsN.shape)


        for i, feat in enumerate(featsN):
            score = score + self.transitions[tagsN[i + 1], tagsN[i]] + featsN[tagsN[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tagsN[-1]]

        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        feats.to(device)
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
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
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
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

    tags2index = {t: i+1 for i, t in enumerate(tags_vocab)}
    tags2index["PAD"] = 0
    ind2tag = {i: w for w, i in tags2index.items()}
    X = [[word2ind[w] for w in s] for s in sentences]
    X = list2tensor(X)
    X = pad_sequence(sequences=X, batch_first=True, padding_value=word2ind["PAD"])
    # Getting labels to pad
    Y = [[tags2index[w] for w in s] for s in tags]
    Y = list2tensor(Y)
    Y = pad_sequence(sequences=Y, batch_first=True, padding_value=tags2index["PAD"])


    return X, Y, word2ind, ind2tag, tags2index, word_vocab,ind2word


x_train, y_train, w2i, i2t, t2i, vocab,i2w = prepare_data('data/train.tsv')

# x_test, y_test, _, _ = prepare_data('data/test.tsv')
# x_dev, y_dev, _, _ = prepare_data('data/dev.tsv')
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
MINIBATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6


train_data = []
for i in range(len(x_train)):
    train_data.append([x_train[i].to(device), y_train[i].to(device)])

trainLoader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32)

i1, l1 = next(iter(trainLoader))

# length of all words 2424757
# t2i = {'PAD': 0, 'ORG': 1, 'PER': 2, 'LOC': 3, 'O': 4, '<START>': 5, '<STOP>': 6}
# len(vocab) = 100841
# length of x_train is:  100000

print(t2i)

t2i[START_TAG] = len(t2i)
t2i[STOP_TAG] = len(t2i)

print(t2i)

model = BiLSTM_CRF(len(w2i), t2i, EMBEDDING_DIM, HIDDEN_DIM).to(device=device)



optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
for epoch in range(1):
    for sentences, tags in trainLoader:
        model.zero_grad()
        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentences, tags)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    print("predictions after training...")
    precheck_sent = prepare_sequence(sentences[0], x_train)
    print(model(precheck_sent))

