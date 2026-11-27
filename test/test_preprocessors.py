import unittest

from pie.data.preprocessors import edit_trees, edit_tree_tuples, greedy_scripts

PAIRS = [("veram", "verus"), ("beatitudinem", "beatitudo"), ("ad", "ad"),
         ("quod", "quod"), ("amabantur", "amo"), ("", "a"), ("abc", "")]


class TestPreprocessorRoundtrip(unittest.TestCase):
    def _check(self, module):
        for tok, lem in PAIRS:
            pred = module.transform(lem, tok)
            self.assertEqual(module.inverse_transform(pred, tok), lem,
                             (tok, lem, pred))

    def test_edit_trees(self):
        self._check(edit_trees)

    def test_greedy_scripts(self):
        for tok, lem in PAIRS:
            if not tok or not lem:  # empty forms never occur in real data
                continue
            pred = greedy_scripts.transform(lem, tok)
            self.assertEqual(greedy_scripts.inverse_transform(pred, tok), lem)

    def test_edit_tree_tuples(self):
        for tok, lem in PAIRS:
            if not tok or not lem:  # tuples need non-empty strings
                continue
            pred = edit_tree_tuples.transform(lem, tok)
            self.assertEqual(edit_tree_tuples.inverse_transform(pred, tok), lem)

    def test_edit_trees_hashable(self):
        # classes are used as vocabulary keys
        hash(edit_trees.transform("verus", "veram"))
