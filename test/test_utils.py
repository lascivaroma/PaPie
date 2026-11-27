import os
import tempfile
import unittest

from pie import utils


class TestUtils(unittest.TestCase):
    def test_text_fns(self):
        self.assertEqual(utils.lower_str("AbC"), "abc")
        self.assertEqual(utils.drop_diacritics("éàü"), "eau")
        self.assertEqual(utils.apply_utfnorm("é", "NFD"), "é")

    def test_compose_and_identity(self):
        self.assertEqual(utils.identity(1), 1)
        self.assertEqual(utils.identity(1, 2), (1, 2))
        fn = utils.compose(utils.lower_str, utils.drop_diacritics)
        self.assertEqual(fn("ÉCOLE"), "ecole")

    def test_window_chunks_flatten(self):
        self.assertEqual(list(utils.window(range(3))),
                         [(None, 0, 1), (0, 1, 2), (1, 2, None)])
        self.assertEqual(list(utils.chunks(range(5), 2)), [[0, 1], [2, 3], [4]])
        self.assertEqual(list(utils.flatten([["ab"], ["c"]])), ["a", "b", "c"])

    def test_recursive_merge(self):
        self.assertEqual(
            utils.recursive_merge({"a": {"b": 1}, "x": 1}, {"a": {"c": 2}, "x": 2}),
            {"a": {"b": 1, "c": 2}, "x": 1})
        self.assertEqual(
            utils.recursive_merge({"x": 1}, {"x": 2}, overwrite=True), {"x": 2})
        with self.assertRaises(ValueError):
            utils.recursive_merge({"a": 1}, {"a": {"b": 1}})

    def test_ensure_ext(self):
        self.assertEqual(utils.ensure_ext("model.pt", "pt", infix="0.87"),
                         "model-0.87.pt")
        self.assertEqual(utils.ensure_ext("model", ".tar"), "model.tar")
        self.assertEqual(utils.ensure_ext("model.test", "pt"), "model.test.pt")

    def test_get_filenames(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("a.tab", "b.tab", ".hidden"):
                open(os.path.join(d, name), "w").close()
            self.assertEqual(len(utils.get_filenames(d)), 2)
            self.assertEqual(utils.get_filenames(os.path.join(d, "a.tab")),
                             [os.path.join(d, "a.tab")])
            self.assertEqual(len(utils.get_filenames(os.path.join(d, "*.tab"))), 2)
