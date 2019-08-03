import torch
from mosi import MOSI
from torch.nn import Module, Embedding, LSTM, Linear, BCELoss, Sigmoid, LayerNorm, BatchNorm1d, Dropout, ReLU
from torch.optim import Adam
from constants import DATA_PATH, WORD_EMB_PATH
from torchtext.data import Field, BucketIterator
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers import ProgressBar


def pp(arr, _):
    for i in range(len(arr)):
        arr[i] = torch.Tensor(arr[i])
    return pad_sequence(arr)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT = Field(lower=True, include_lengths=True)
VISUAL = Field(sequential=False, use_vocab=False, postprocessing=pp, dtype=torch.float)
ACOUSTIC = Field(sequential=False, use_vocab=False, postprocessing=pp, dtype=torch.float)
LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)

train, dev, test = MOSI.splits(DATA_PATH, TEXT, VISUAL, ACOUSTIC, LABEL)
# delete neural cases cause ratio (neg: 44%, pos: 52%, neu: 4%)
for dataset in (train, dev, test):
    new_examples = []
    for example in dataset.examples:
        if example.label == 0.0:
            continue
        elif example.label < 0.0:
            example.label = 0.0
        else:
            example.label = 1.0
        new_examples.append(example)
    dataset.examples = new_examples

TEXT.build_vocab(train, vectors='glove.42B.300d', vectors_cache=WORD_EMB_PATH)

train_iter, dev_iter, test_iter = BucketIterator.splits(
    (train, dev, test), batch_sizes=(64, 64 * 3, 64 * 3), sort_key=lambda ex: len(ex.text),
    device='cuda', sort_within_batch=True)


class Model(Module):
    def __init__(self, vocab_size, text_dim, input_sizes, hidden_sizes, fc1_size, dropout_rate):
        super().__init__()
        self.emb = Embedding(vocab_size, text_dim)
        self.emb.weight.data = TEXT.vocab.vectors
        self.emb.weight.data.requires_grd = False
        self.trnn1 = LSTM(input_sizes[0], hidden_sizes[0], bidirectional=True)
        self.trnn2 = LSTM(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)

        self.vrnn1 = LSTM(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = LSTM(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)

        self.arnn1 = LSTM(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = LSTM(2 * hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        self.tlayer_norm = LayerNorm([hidden_sizes[0] * 2])
        self.vlayer_norm = LayerNorm([hidden_sizes[1] * 2])
        self.alayer_norm = LayerNorm([hidden_sizes[2] * 2])
        self.bn = BatchNorm1d(sum(hidden_sizes) * 4)
        self.dropout = Dropout(dropout_rate)
        self.fc1 = Linear(sum(hidden_sizes) * 4, fc1_size)
        self.fc2 = Linear(fc1_size, 1)
        self.sig = Sigmoid()

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths)
        packed_h1, (final_h1, _) = rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)
        _, (final_h2, _) = rnn2(packed_normed_h1)
        return final_h1, final_h2

    def fusion(self, text, visual, acoustic, lengths):
        batch_size = lengths.size(0)

        # extract features from text modality
        final_h1t, final_h2t = self.extract_features(text, lengths, self.trnn1, self.trnn2, self.tlayer_norm)

        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)

        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)

        # simple late fusion -- concatenation + normalization
        h = torch.cat((final_h1t, final_h2t, final_h1v, final_h2v, final_h1a, final_h2a),
                      dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        return self.bn(h)

    def forward(self, text, visual, acoustic):
        index = text[0]
        length = text[1]
        text = self.emb(index)
        x = self.fusion(text, visual, acoustic, length)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        y = self.sig(x).squeeze()
        return y


vocab_size, text_dim = TEXT.vocab.vectors.shape
text_size, visual_size, acoustic_size = 300, 47, 74
dropout = 0.3
input_sizes = [text_size, visual_size, acoustic_size]
hidden_sizes = [int(text_size*1.5), int(visual_size*1.5), int(acoustic_size*1.5)]
fc1_size = sum(hidden_sizes) // 2
model = Model(vocab_size, text_dim, input_sizes, hidden_sizes, fc1_size, dropout)
model.to(device)
optimizer = Adam(model.parameters(), lr=6e-4)
loss_fn = BCELoss()

if __name__ == '__main__':
    def update(engine, batch):
        model.train()
        optimizer.zero_grad()
        text, visual, acoustic = batch.text, batch.visual, batch.acoustic
        y = batch.label
        y_pred = model(text, visual, acoustic)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()


    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            text, visual, acoustic = batch.text, batch.visual, batch.acoustic
            y = batch.label
            y_pred = model(text, visual, acoustic)
            return y_pred, y


    trainer = Engine(update)
    train_evaluator = Engine(inference)
    validation_evaluator = Engine(inference)
    # add metric
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')  # for pbar
    Accuracy().attach(train_evaluator, 'acc')
    Loss(loss_fn).attach(train_evaluator, 'bce')
    Accuracy().attach(validation_evaluator, 'acc')
    Loss(loss_fn).attach(validation_evaluator, 'bce')

    # add Progress Bar
    pbar = ProgressBar(persist=True, bar_format='')
    pbar.attach(trainer, ['loss'])

    # add early stopping
    def score_fn_1(engine):
        val_loss = engine.state.metrics['bce']
        return -val_loss


    early_stop = EarlyStopping(patience=10, score_function=score_fn_1, trainer=trainer)
    validation_evaluator.add_event_handler(Events.COMPLETED, early_stop)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_iter)
        metrics = train_evaluator.state.metrics
        accuracy = metrics['acc']
        bce = metrics['bce']
        pbar.log_message(
            f'Training Results - Epoch: {engine.state.epoch} Avg accuracy: {accuracy} Avg loss: {bce}')


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        validation_evaluator.run(dev_iter)
        metrics = validation_evaluator.state.metrics
        accuracy = metrics['acc']
        bce = metrics['bce']
        pbar.log_message(
            f'Validation Results - Epoch: {engine.state.epoch} Avg accuracy: {accuracy} Avg loss: {bce}')

    # add Checkpointer
    def score_fn_2(engine):
        return engine.state.metrics['acc']


    check_point = ModelCheckpoint('./models', 'avg_concat', score_function=score_fn_2, score_name='acc',
                                  require_empty=False, save_as_state_dict=True)
    validation_evaluator.add_event_handler(Events.COMPLETED, check_point, {'model': model})

    trainer.run(train_iter, max_epochs=200)
