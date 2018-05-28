
import os
import glob
import time
import json
import logging
from collections import Counter, defaultdict
import random

import torch

from pie import utils, torch_utils, constants


class LineParseException(Exception):
    pass


class BaseReader(object):
    """
    Abstract reader class

    Parameters
    ==========
    input_path : str (optional), either a path to a directory, a path to a file
        or a unix style pathname pattern expansion for glob

    Settings
    ========
    shuffle : bool, whether to shuffle files after each iteration
    """
    def __init__(self, settings, input_path=None):
        input_path = input_path or settings.input_path

        if os.path.isdir(input_path):
            self.filenames = [os.path.join(input_path, f)
                              for f in os.listdir(input_path)
                              if not f.startswith('.')]
        elif os.path.isfile(input_path):
            self.filenames = [input_path]
        else:
            self.filenames = glob.glob(input_path)

        if len(self.filenames) == 0:
            raise ValueError("Couldn't find files in {}".format(input_path))

        # settings
        self.shuffle = settings.shuffle

    def reset(self):
        """
        Called after a full run over `readsents`
        """
        if self.shuffle:
            random.shuffle(self.filenames)

    def readsents(self):
        """
        Generator over dataset sentences. Each output is a tuple of (Input, Tasks)
        objects with, where each entry is a list of strings.
        """
        current_sent = 0

        for fpath in self.filenames:
            lines = self.parselines(fpath, self.get_tasks(fpath))

            while True:
                try:
                    yield (fpath, current_sent), next(lines)
                    current_sent += 1

                except LineParseException as e:
                    logging.warning(
                        "Parse error at [{}:sent={}]\n  => {}"
                        .format(fpath, current_sent + 1, str(e)))
                    continue

                except StopIteration:
                    break

        self.reset()

    def check_tasks(self, expected=None):
        """
        Returns the union of the tasks across files. If `expected` is passed,
        it is understood to be a subset of all available tasks and it will check
        whether all `expected` are in fact available in all files as per the output
        of `get_tasks`.

        Returns
        ========
        List of tasks available in all files (or a subset if `expected` is passed)
        """
        tasks = None
        for fpath in self.filenames:
            if tasks is None:
                tasks = set(self.get_tasks(fpath))
            else:
                tasks = tasks.intersection(set(self.get_tasks(fpath)))

        if expected is not None:
            diff = set(expected).difference(tasks)
            if diff:
                raise ValueError("Following expected tasks are missing "
                                 "from at least one file: '{}'"
                                 .format('"'.join(diff)))

        return expected or list(tasks)

    def parselines(self, fpath, tasks):
        """
        Generator over tuples of (Input, Tasks) holding input and target data
        as lists of strings. Some tasks might be missing from a file, in which
        case the corresponding value is None.

        ParseError can be thrown if an issue is encountered during processing.
        The issue can be documented by passing an error message as second argument
        to ParseError
        """
        raise NotImplementedError

    def get_tasks(self, fpath):
        """
        Reader is responsible for extracting the expected tasks in the file.

        Returns
        =========
        Set of tasks in a file
        """
        raise NotImplementedError


class LineParser(object):
    """
    Inner class to handle sentence breaks
    """
    def __init__(self, tasks, breakline_type, breakline_ref, breakline_data):
        # breakline info
        self.breakline_type = breakline_type
        self.breakline_ref = breakline_ref
        self.breakline_data = breakline_data
        # data
        self.inp = []
        self.tasks = {task: [] for task in tasks}

    def add(self, line, linenum):
        """
        Adds line to current sentence.
        """
        inp, *tasks = line.split()
        if len(tasks) != len(self.tasks):
            raise LineParseException(
                "Not enough number of tasks. Expected {} but got {} at line {}."
                .format(len(self.tasks), len(tasks), linenum))

        self.inp.append(inp)

        for task, data in zip(self.tasks.keys(), tasks):
            try:
                data = getattr(self, 'process_{}'.format(task.lower()))(data)
            except AttributeError:
                pass
            finally:
                self.tasks[task].append(data)

    def check_breakline(self):
        """
        Check if sentence is finished.
        """
        if self.breakline_ref == 'input':
            ref = self.inp
        else:
            ref = self.tasks[self.breakline_ref]

        if self.breakline_type == 'FULLSTOP':
            if ref[-1] == self.breakline_data:
                return True
        elif self.breakline_type == 'LENGTH':
            if len(ref) == self.breakline_data:
                return True

    def reset(self):
        """
        Reset sentence data
        """
        self.tasks, self.inp = {task: [] for task in self.tasks}, []


class CustomLineParser(LineParser):
    # TODO: parse morphology into some data structure
    def process_morph(self, data):
        pass


class TabReader(BaseReader):
    """
    Reader for files in tab format where each line has annotations for a given word
    and each annotation is located in a column separated by tabs.

    ...
    italiae	italia	NE	gender=FEMININE|case=GENITIVE|number=SINGULAR
    eo	is	PRO	gender=MASCULINE|case=ABLATIVE|number=SINGULAR
    tasko	taskus	NN	gender=MASCULINE|case=ABLATIVE|number=SINGULAR
    ...

    Settings
    ===========
    breakline_type : str, one of "LENGTH" or "FULLSTOP".
    breakline_data : str or int, if breakline_type is LENGTH it will be assumed
        to be an integer defining the number of words per sentence, and the
        dataset will be break into equally sized chunks. If breakline_type is
        FULLSTOP it will be assumed to be a POS tag to use as criterion to
        split sentences.
    """
    def __init__(self, settings, line_parser=LineParser, **kwargs):
        super(TabReader, self).__init__(settings, **kwargs)

        self.line_parser = line_parser
        self.breakline_type = settings.breakline_type
        self.breakline_ref = settings.breakline_ref
        self.breakline_data = settings.breakline_data
        self.max_sent_len = settings.max_sent_len
        self.tasks_order = settings.tasks_order

    def parselines(self, fpath, tasks):
        """
        Generator over sentences in a single file
        """
        with open(fpath, 'r+') as f:
            parser = self.line_parser(
                tasks, self.breakline_type, self.breakline_ref, self.breakline_data)

            for line_num, line in enumerate(f):
                line = line.strip()

                if not line:    # avoid empty line
                    continue

                parser.add(line, line_num)

                if parser.check_breakline() or len(parser.inp) >= self.max_sent_len:
                    yield parser.inp, parser.tasks
                    parser.reset()

    def get_tasks(self, fpath):
        """
        Guess tasks from file assuming expected order
        """
        with open(fpath, 'r+') as f:

            # move to first non empty line
            line = next(f).strip()
            while not line:
                line = next(f).strip()

            _, *tasks = line.split()
            if len(tasks) == 0:
                raise ValueError("Not enough input tasks: [{}]".format(fpath))

            return self.tasks_order[:len(tasks)]


class LabelEncoder(object):
    """
    Label encoder
    """
    def __init__(self, pad=True, eos=True, bos=False,
                 vocabsize=None, level='word', name='Unk'):

        if level.lower() not in ('word', 'char'):
            raise ValueError("`level` must be 'word' or 'char'")

        self.eos = constants.EOS if eos else None
        self.pad = constants.PAD if pad else None
        self.bos = constants.BOS if bos else None
        self.vocabsize = vocabsize
        self.level = level.lower()
        self.name = name
        self.reserved = (constants.UNK,)  # always use <unk>
        self.reserved += tuple([sym for sym in [self.eos, self.pad, self.bos] if sym])
        self.freqs = Counter()
        self.known_tokens = set()  # for char-level dicts, keep word-level known tokens
        self.table = None
        self.inverse_table = None
        self.fitted = False

    def __len__(self):
        if not self.fitted:
            raise ValueError("Cannot get length of unfitted LabelEncoder")
        return len(self.table)

    def __eq__(self, other):
        if type(other) != LabelEncoder:
            return False

        return self.pad == other.pad and \
            self.eos == other.eos and \
            self.bos == other.bos and \
            self.vocabsize == other.vocabsize and \
            self.level == other.level and \
            self.freqs == other.freqs and \
            self.table == other.table and \
            self.inverse_table == other.inverse_table and \
            self.fitted == other.fitted

    def add(self, sent):
        if self.fitted:
            raise ValueError("Already fitted")

        if self.level == 'word':
            self.freqs.update(sent)
        else:
            self.freqs.update(utils.flatten(sent))
            self.known_tokens.update(sent)

    def compute_vocab(self):
        if self.fitted:
            raise ValueError("Cannot compute vocabulary, already fitted")

        if len(self.freqs) == 0:
            logging.warning("Computing vocabulary for empty encoder {}"
                            .format(self.name))

        most_common = self.freqs.most_common(n=self.vocabsize or len(self.freqs))
        self.inverse_table = list(self.reserved) + [sym for sym, _ in most_common]
        self.table = {sym: idx for idx, sym in enumerate(self.inverse_table)}
        self.fitted = True

    def transform(self, sent):
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        sent = [self.table.get(tok, self.table[constants.UNK]) for tok in sent]

        if self.eos:
            sent.append(self.get_eos())

        return sent

    def inverse_transform(self, sent):
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        return [self.inverse_table[i] for i in sent]

    def stringify(self, sent):
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        try:
            sent = sent[:sent.index(self.get_eos())]
        except ValueError:
            pass

        return self.inverse_transform(sent)

    def _get_sym(self, sym):
        if not self.fitted:
            raise ValueError("Vocabulary hasn't been computed yet")

        return self.table.get(sym)

    def get_pad(self):
        return self._get_sym(constants.PAD)

    def get_eos(self):
        return self._get_sym(constants.EOS)

    def get_bos(self):
        return self._get_sym(constants.BOS)

    def jsonify(self):
        if not self.fitted:
            raise ValueError("Attempted to serialize unfitted encoder")

        return {'eos': self.eos,
                'pad': self.pad,
                'level': self.level,
                'vocabsize': self.vocabsize,
                'freqs': dict(self.freqs),
                'table': dict(self.table),
                'inverse_table': self.inverse_table,
                'known_tokens': list(self.known_tokens)}

    @classmethod
    def from_json(cls, obj):
        inst = cls(pad=obj['pad'], eos=obj['eos'],
                   level=obj['level'], vocabsize=obj['vocabsize'])
        inst.freqs = Counter(obj['freqs'])
        inst.table = dict(obj['table'])
        inst.inverse_table = list(obj['inverse_table'])
        inst.known_tokens = set(obj['known_tokens'])
        inst.fitted = True

        return inst


class MultiLabelEncoder(object):
    """
    Complex Label encoder for all tasks.
    """
    def __init__(self, word_vocabsize=None, char_vocabsize=None):
        self.word = LabelEncoder(vocabsize=word_vocabsize, name='word')
        self.char = LabelEncoder(vocabsize=char_vocabsize, name='char', level='char')
        self.insts = 0
        self.tasks = {}

    def add_task(self, name, **kwargs):
        self.tasks[name] = LabelEncoder(name=name, **kwargs)
        return self

    @classmethod
    def from_settings(cls, settings):
        return cls(word_vocabsize=settings.word_vocabsize,
                   char_vocabsize=settings.char_vocabsize)

    def fit(self, lines):
        """
        Parameters
        ===========
        lines : iterator over tuples of (Input, Tasks)
        """
        for idx, (inp, tasks) in enumerate(lines):
            # increment counter
            self.insts += 1
            # input
            self.word.add(inp)
            self.char.add(inp)

            for task, le in self.tasks.items():
                le.add(tasks[task])

        self.word.compute_vocab()
        self.char.compute_vocab()
        for le in self.tasks.values():
            le.compute_vocab()

        return self

    def transform(self, sents):
        """
        Parameters
        ===========
        sents : list of Example's

        Returns
        ===========
        tuple of (word, char), task_dict

            - word: list of integers
            - char: list of integers where each list represents a word at the
                character level
            - task_dict: Dict to corresponding integer output for each task
        """
        word, char, tasks_dict = [], [], defaultdict(list)

        for inp, tasks in sents:
            # input data
            word.append(self.word.transform(inp))
            for w in inp:
                char.append(self.char.transform(w))

            # task data
            for task, le in self.tasks.items():
                if le.level == 'word':
                    tasks_dict[task].append(le.transform(tasks[task]))
                else:
                    for w in tasks[task]:
                        tasks_dict[task].append(le.transform(w))

        return (word, char), tasks_dict

    def save(self, path):
        with open(path, 'w+') as f:
            obj = {'word': self.word.jsonify(),
                   'char': self.char.jsonify(),
                   'insts': self.insts,
                   'tasks': {le.name: le.jsonify() for le in self.tasks.values()}}
            json.dump(obj, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r+') as f:
            obj = json.load(f)

        inst = cls()  # dummy instance to overwrite

        inst.insts = obj['insts']
        inst.word = LabelEncoder.from_json(obj['word'])
        inst.char = LabelEncoder.from_json(obj['char'])

        for task, le_obj in obj['tasks'].items():
            inst.tasks[task] = LabelEncoder.from_json(le_obj)

        return inst


class Dataset(object):
    """
    Dataset class to encode files into integers and compute batches.

    Settings
    ===========
    buffer_size : int, maximum number of sentences in memory at any given time.
       The larger the buffer size the more memory instensive the dataset will
       be but also the more effective the shuffling over instances.
    batch_size : int, number of sentences per batch
    device : str, target device to put the processed batches on
    shuffle : bool, whether to shuffle items in the buffer

    Parameters
    ===========
    label_encoder : optional, prefitted LabelEncoder object
    """
    def __init__(self, settings, reader=None, label_encoder=None, expected_tasks=None):
        if settings.batch_size > settings.buffer_size:
            raise ValueError("Not enough buffer capacity {} for batch_size of {}"
                             .format(settings.buffer_size, settings.batch_size))

        # attributes
        self.buffer_size = settings.buffer_size
        self.batch_size = settings.batch_size
        self.device = settings.device
        self.shuffle = settings.shuffle

        # data
        self.dev_sents = defaultdict(set)
        # TODO: this assumes TabReader. In the future we would need one per file
        self.reader = reader or TabReader(settings)
        tasks = self.reader.check_tasks(expected=expected_tasks)
        # label encoder
        if label_encoder is None:
            label_encoder = MultiLabelEncoder.from_settings(settings)
            for task in tasks:
                label_encoder.add_task(task, **settings.tasks[task])
            if settings.verbose:
                print("::: Fitting data... :::")
                print()
            start = time.time()
            label_encoder.fit(line for _, line in self.reader.readsents())
            if settings.verbose:
                print("Done in {:g} secs".format(time.time() - start))
                print()
        if settings.verbose:
            print("::: Available tasks :::")
            print()
            for task in tasks:
                print("- {}".format(task))
            print()
        self.label_encoder = label_encoder

        if len(self) <= 0:
            raise ValueError("Not enough instances [{}] in dataset".format(len(self)))

    def __len__(self):
        dev_sents = sum(len(v) for v in self.dev_sents.values())
        return (self.label_encoder.insts - dev_sents) // self.batch_size

    @staticmethod
    def get_nelement(batch):
        """
        Returns the number of elements in a batch (based on word-level length)
        """
        return batch[0][0][1].sum().item()

    def pack_batch(self, batch, device=None):
        """
        Transform batch data to tensors
        """
        device = device or self.device

        (word, char), tasks = self.label_encoder.transform(batch)

        word = torch_utils.pad_batch(
            word, self.label_encoder.word.get_pad(), device=device)
        char = torch_utils.pad_batch(
            char, self.label_encoder.char.get_pad(), device=device)

        output_tasks = {}
        for task, data in tasks.items():
            output_tasks[task] = torch_utils.pad_batch(
                data, self.label_encoder.tasks[task].get_pad(),
                device=device)

        return (word, char), output_tasks

    def prepare_buffer(self, buf, **kwargs):
        "Transform buffer into batch generator"

        def key(data):
            inp, tasks = data
            return len(inp)

        buf = sorted(buf, key=key, reverse=True)
        batches = list(utils.chunks(buf, self.batch_size))

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield self.pack_batch(batch, **kwargs)

    def get_dev_split(self, split=0.05):
        "Grab a dev split from the dataset"
        if len(self.dev_sents) > 0:
            raise RuntimeError("Dev-split already exists! Cannot create a new one")

        buf = []
        sents = self.reader.readsents()

        while len(buf) < (self.label_encoder.insts * split):
            (fpath, line_num), data = next(sents)
            buf.append(data)
            self.dev_sents[fpath].add(line_num)

        # get batches on cpu
        batches = list(self.prepare_buffer(buf, device='cpu'))

        return device_wrapper(batches, device=self.device)

    def batch_generator(self):
        """
        Generator over dataset batches. Each batch is a tuple of (input, tasks):
            * (word, char)
                - word : tensor(length, batch_size), padded lengths
                - char : tensor(length, batch_size * words), padded lengths
            * (tasks) dictionary with tasks
        """
        buf = []
        for (fpath, line_num), data in self.reader.readsents():

            # check if buffer is full and yield
            if len(buf) == self.buffer_size:
                yield from self.prepare_buffer(buf)
                buf = []

            # don't use dev sentences
            if fpath in self.dev_sents and line_num in self.dev_sents[fpath]:
                continue

            # fill buffer
            buf.append(data)

        if len(buf) > 0:
            yield from self.prepare_buffer(buf)


def wrap_device(it, device):
    for i in it:
        if isinstance(i, torch.Tensor):
            yield i.to(device)
        elif isinstance(i, dict):
            yield {k: tuple(wrap_device(v, device)) for k, v in i.items()}
        else:
            yield tuple(wrap_device(i, device))


class device_wrapper(object):
    def __init__(self, batches, device):
        self.batches = batches
        self.device = device

    def __getitem__(self, idx):
        return tuple(wrap_device(self.batches[idx], self.device))

    def __len__(self):
        return len(self.batches)
