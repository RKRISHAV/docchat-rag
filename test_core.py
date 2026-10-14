import unittest
from unittest.mock import patch
from pathlib import Path

# The UI app executes Streamlit at import time; test its pure helpers by extracting definitions.
import ast
source = Path(__file__).resolve().parents[1].joinpath('app.py').read_text()
module = ast.parse(source)
functions = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in ('chunk_text', 'retrieve')]
ns = {'re': __import__('re'), 'cosine_similarity': __import__('sklearn.metrics.pairwise', fromlist=['cosine_similarity']).cosine_similarity}
exec(compile(ast.Module(body=functions, type_ignores=[]), '<helpers>', 'exec'), ns)

class CoreTests(unittest.TestCase):
    def test_chunking(self):
        chunks = ns['chunk_text']('alpha beta gamma ' * 200, 100, 20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 100 for c in chunks))
    def test_empty(self):
        self.assertEqual(ns['chunk_text']('   '), [])
    def test_retrieval(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        passages = [{'text':'the sun is a star', 'source':'a'}, {'text':'python code functions', 'source':'b'}]
        v = TfidfVectorizer().fit([p['text'] for p in passages])
        results = ns['retrieve']('python functions', passages, v, v.transform([p['text'] for p in passages]), 1)
        self.assertEqual(results[0][0]['source'], 'b')

if __name__ == '__main__':
    unittest.main()
