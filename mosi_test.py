import os
import torch

from ignite.engine import Engine
from ignite.metrics import Accuracy

from mosi_model import model, test_iter


# get test result for each saved best model
def log_test_results(model, test_iter):
    # create test engine
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            text, visual, acoustic = batch.text, batch.visual, batch.acoustic
            y = batch.label
            y_pred = model(text, visual, acoustic)
            return y_pred, y

    test_evaluator = Engine(inference)
    # add metrics accuracy
    Accuracy().attach(test_evaluator, 'acc')
    # run engine and get accuracy
    test_evaluator.run(test_iter)
    accuracy = test_evaluator.state.metrics['acc']
    print(f'Test Results - Accuracy: {accuracy}')
    return accuracy


accs = []
for pth in os.listdir('./models'):
    pth = os.path.join('./models', pth)
    model.load_state_dict(torch.load(pth))
    acc = log_test_results(model, test_iter)
    accs.append(acc)
print(f'Avg accuracy: {sum(accs)/len(accs)}')

