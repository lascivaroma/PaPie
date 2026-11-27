import os
import tempfile
import unittest

import torch

from pie import tagger
from pie.data import MultiLabelEncoder, Reader
from pie.models import SimpleModel
from pie.settings import settings_from_file

testpath = os.path.join(os.path.dirname(__file__), 'models', 'testconfig.json')


class TestSentenceReaders(unittest.TestCase):
    def _write(self, d, text):
        path = os.path.join(d, "in.txt")
        with open(path, "w") as f:
            f.write(text)
        return path

    def test_line_reader_splits_on_max_len(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, "a b c d e\nf g\n")
            sents = list(tagger.get_sentences_line(path, 3))
            self.assertEqual(sents, [["a", "b", "c"], ["d", "e"], ["f", "g"]])

    def test_vrt_reader(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, "a x\nb x\n\nc x\n")
            self.assertEqual(list(tagger.get_sentences_vrt(path, 10)),
                             [["a", "b"], ["c"]])

    def test_simple_tokenizer(self):
        sents = list(tagger.simple_tokenizer("Hello, world. Bye."))
        self.assertEqual(sents, [["Hello", ",", "world", "."], ["Bye", "."]])


class TestTagger(unittest.TestCase):
    """End to end: build a tiny untrained model, save it, tag with it"""

    @classmethod
    def setUpClass(cls):
        settings = settings_from_file(testpath)
        le = MultiLabelEncoder.from_settings(settings)
        le.fit_reader(Reader(settings, settings.input_path))
        torch.manual_seed(0)
        model = SimpleModel(le, settings.tasks, 16, 16, 16, 1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "model")
        model.save(cls.path)
        cls.tasks = [t["name"] for t in settings.tasks]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _tag(self, **kw):
        t = tagger.Tagger(**kw)
        t.add_model(self.path + ".tar", *self.tasks)
        sents = [["quod", "ad", "veram"], ["beatitudinem"]]
        return t.tag(sents, [3, 1])

    def test_tag_shapes(self):
        tagged, tasks = self._tag()
        self.assertEqual([len(s) for s in tagged], [3, 1])
        self.assertEqual(tagged[0][0][0], "quod")
        self.assertEqual(len(tagged[0][0][1]), len(tasks))

    def test_quantize_and_cache_run(self):
        # deprecated upstream in recent torch, but must keep working
        tagged, _ = self._tag(quantize=True, cache=True)
        self.assertEqual([len(s) for s in tagged], [3, 1])

    def test_unknown_task(self):
        t = tagger.Tagger()
        with self.assertRaises(ValueError):
            t.add_model(self.path + ".tar", "nonexistent")

    def test_deterministic(self):
        self.assertEqual(self._tag()[0], self._tag()[0])


class TestTokenizeFile(unittest.TestCase):
    def test_lines_from_file_tokenize(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "in.txt")
            with open(path, "w") as f:
                f.write("Hello, world. Bye.\nAgain.\n")
            sents = [s for s, _ in tagger.lines_from_file(path, tokenize=True)]
            self.assertEqual(sents[0], ["Hello", ",", "world", "."])
            self.assertEqual(len(sents), 3)

    def test_simple_tokenizer_lower(self):
        self.assertEqual(list(tagger.simple_tokenizer("Hi, You.", lower=True)),
                         [["hi", ",", "you", "."]])
