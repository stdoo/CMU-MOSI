import pickle
from torchtext.data import Dataset, Example


class MOSI(Dataset):

    name = 'cmu_mosi'
    dirname = name

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, visual_field, acoustic_field, label_field, **kwargs):
        """Create an MOSI dataset instance given a path and fields.

        Arguments:
            path: Path to the dataset's highest level directory
            text_field: The field that will be used for text data.
            visual_field: The field that will be used for visual data.
            acoustic_field: The field that will be used for acoustic data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('visual', visual_field), ('acoustic', acoustic_field), ('label', label_field)]
        examples = []

        with open(path, 'rb') as f:
            data = pickle.load(f)
        for ex in data:
            (text, visual, acoustic), label, _ = ex
            examples.append(Example.fromlist([text, visual, acoustic, label], fields))

        super(MOSI, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path, text_field, visual_field, acoustic_field, label_field,
               train='train', validation='dev', test='test', **kwargs):
        """Create dataset objects for splits of the MOSI dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            visual_field: The field that will be used for visual data.
            acoustic_field: The field that will be used for acoustic data.
            label_field: The field that will be used for label data.
            root: Root dataset storage directory. Default is '.data'.
            train: The directory that contains the training examples
            test: The directory that contains the test examples
            validation: The directory that contains the dev examples
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        return super(MOSI, cls).splits(
            path=path, text_field=text_field, visual_field=visual_field, acoustic_field=acoustic_field,
            label_field=label_field, train=train, validation=validation, test=test, **kwargs)

