import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import load_data,tokenize
from demo import app,infer

class SystemTests(unittest.TestCase):
    def test_split(self):
        df,s=load_data(); self.assertEqual(sum(len(s[k]) for k in ['train','val','test']),len(df)); self.assertFalse(set(s['train'])&set(s['test']))
    def test_tokenization(self): self.assertEqual(tokenize("Team's WIN 123!"),["team's","win"])
    def test_empty(self): self.assertEqual(app.test_client().post('/api/predict',json={'text':' '}).status_code,400)
    def test_non_object(self): self.assertEqual(app.test_client().post('/api/predict',json=[]).status_code,400)
    def test_invalid_model(self):
        for value in ['bad',{},[],None,1]:
            self.assertEqual(app.test_client().post('/api/predict',json={'text':'football','model':value}).status_code,400)
    def test_too_long(self): self.assertEqual(app.test_client().post('/api/predict',json={'text':'a'*30001}).status_code,400)
    def test_oov(self): self.assertTrue(infer('中文测试 12345')['abstain'])
    def test_probabilities(self): self.assertAlmostEqual(sum(infer('The football team won the championship.')['probabilities'].values()),1.)
    def test_explanation(self):
        d=infer('The football team won the championship after the coach changed the players.'); self.assertAlmostEqual(d['margin'],d['reconstructed_margin'],places=8)
    def test_api_and_page(self):
        c=app.test_client(); self.assertEqual(c.get('/').status_code,200); self.assertEqual(c.get('/api/results').status_code,200); self.assertEqual(c.get('/api/health').json['status'],'ok')

if __name__=='__main__': unittest.main(verbosity=2)
