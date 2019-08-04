import torch
from mosi import MOSI
from torch.nn import Module, Embedding, Linear, BCELoss, Sigmoid
from torch.optim import Adam
from constants import DATA_PATH, WORD_EMB_PATH
from torchtext.data import Field, BucketIterator
from torch.nn.utils.rnn import pad_sequence

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers import ProgressBar


def pp(arr, _):
    for i in range(len(arr)):
        arr[i] = torch.Tensor(arr[i])
    return pad_sequence(arr, batch_first=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT = Field(lower=True, batch_first=True)
VISUAL = Field(sequential=False, use_vocab=False, postprocessing=pp, dtype=torch.float, batch_first=True)
ACOUSTIC = Field(sequential=False, use_vocab=False, postprocessing=pp, dtype=torch.float, batch_first=True)
LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float, batch_first=True)

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
    def __init__(self):
        super().__init__()
        # self.emb = Embedding(vocab_size, text_dim)
        # self.emb.weight.data = TEXT.vocab.vectors
        # self.emb.weight.data.requires_grd = False
        self.fc = Linear(74, 1)
        self.sig = Sigmoid()

    def forward(self, acoustic):
        # text = self.emb(text)
        # text = torch.mean(text, 1)
        acoustic = torch.mean(acoustic, 1)
        # acoustic = torch.mean(acoustic, 1)
        #
        # x = torch.cat((text, visual, acoustic), 1)
        y = self.fc(acoustic)
        y = self.sig(y).squeeze()
        return y


vocab_size, text_dim = TEXT.vocab.vectors.shape
model = Model()
model.to(device)
optimizer = Adam(model.parameters(), lr=6e-3)
loss_fn = BCELoss()

if __name__ == '__main__':
    def update(engine, batch):
        model.train()
        optimizer.zero_grad()
        text, visual, acoustic = batch.text, batch.visual, batch.acoustic
        y = batch.label
        y_pred = model(acoustic)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()


    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            text, visual, acoustic = batch.text, batch.visual, batch.acoustic
            y = batch.label
            y_pred = model(acoustic)
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
