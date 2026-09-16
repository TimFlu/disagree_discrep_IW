import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.lib.source_restriction import (restriction_splits, source_mask, dis2_components,
    evaluate_source_restriction, REFERENCE)
from src.lib.hybrid_critic import train_hybrid_critic
from src.eval.source_restriction import main as evaluate_cli
from src.plot.source_restriction import main as plot_cli


class FixedDomain:
    def __init__(self, scores): self.scores=scores
    def log_odds(self, x): return torch.logit(torch.as_tensor(self.scores,dtype=torch.float64))
    def state_dict(self): return {}


def data(n=100, nt=120):
    torch.manual_seed(4)
    xs,xt=torch.randn(n,4),torch.randn(nt,4)+.25
    sl=torch.stack([-xs[:,0],xs[:,0]],1)
    tl=torch.stack([-xt[:,0],xt[:,0]],1)
    sy=(xs[:,1]>0).long()
    return xs,sl,sy,xt,tl


class SourceRestrictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_threshold_direction_and_boundaries(self):
        np.testing.assert_array_equal(source_mask([.1,.5,.9],.5),[False,True,True])
        self.assertTrue(source_mask([0.,1.],0).all())
        with self.assertRaises(ValueError): source_mask([.5],1.1)

    def test_splits_reuse_and_heldout_partition(self):
        split=restriction_splits(101,7)
        self.assertEqual([len(split[k]) for k in split],[40,10,51])
        self.assertEqual(len(np.unique(np.concatenate(list(split.values())))),101)
        for key in split: np.testing.assert_array_equal(split[key],restriction_splits(101,7)[key])

    def test_formula_uses_passed_source_count_and_no_target_mass(self):
        result=dis2_components([1,0],[0,0],[1,0,1,0],delta=.05)
        self.assertEqual(result['error_estimate_raw'],1.)
        self.assertEqual(result['n_val_source'],2)
        expected=np.sqrt((2+4*4)*np.log(20)/(2*2*4))
        self.assertAlmostEqual(result['epsilon'],expected)
        full=dis2_components([1,0,0,0],[0,0,0,0],[1,0,1,0],delta=.05)
        self.assertGreater(result['epsilon'],full['epsilon'])

    def run_fixed(self, scores, **kwargs):
        with patch('src.lib.source_restriction.fit_domain',return_value=(FixedDomain(scores),{})):
            return evaluate_source_restriction(*data(),domain_epochs=2,epochs=3,repeats=2,batch_size=17,**kwargs)

    def test_zero_threshold_matches_full_dis2_reference(self):
        rows,_,_=self.run_fixed(np.linspace(.1,.9,100),methods=['source_pruning'],thresholds=[0.],include_reference=True)
        for key in ['source_error','source_disagreement','target_disagreement','epsilon','lower_bound','critic_epoch']:
            self.assertAlmostEqual(rows[0][key],rows[1][key])

    def test_constant_soft_weights_match_reference_and_full_counts(self):
        rows,_,_=self.run_fixed(np.full(100,.5),methods=['source_weighted_critic'],include_reference=True)
        self.assertEqual(len(rows),2)  # Soft method is not repeated for each hard threshold.
        for key in ['error_estimate_raw','epsilon','lower_bound']:
            self.assertAlmostEqual(rows[0][key],rows[1][key])
        self.assertEqual(rows[0]['n_val_source'],50)
        self.assertEqual(rows[0]['n_val_target'],60)

    def test_pruning_applied_to_all_source_splits_never_target(self):
        scores=np.linspace(.05,.95,100)
        calls=[]
        def spy(*args,**kwargs):
            calls.append(args)
            return train_hybrid_critic(*args,**kwargs)
        with patch('src.lib.source_restriction.train_hybrid_critic',side_effect=spy):
            rows,splits,artifacts=self.run_fixed(scores,methods=['source_pruning'],thresholds=[.5])
        row=rows[0]
        for i,name in [(0,'fit'),(2,'validation')]:
            self.assertEqual(len(calls[0][i][0]),int((scores[splits['source'][name]]>=.5).sum()))
            self.assertEqual(len(calls[0][i+1][0]),len(splits['target'][name]))
            self.assertTrue((calls[0][i][2]==1).all())
        self.assertEqual(row['n_val_source'],int((scores[splits['source']['eval']]>=.5).sum()))
        self.assertEqual(row['n_val_target'],len(splits['target']['eval']))
        np.testing.assert_array_equal(artifacts['source_masks']['tau0.5'],scores>=.5)

    def test_soft_weights_only_enter_training_and_selection(self):
        scores=np.linspace(.05,.95,100)
        calls=[]
        def spy(*args,**kwargs):
            calls.append(args)
            return train_hybrid_critic(*args,**kwargs)
        with patch('src.lib.source_restriction.train_hybrid_critic',side_effect=spy):
            rows,splits,artifacts=self.run_fixed(scores,methods=['source_weighted_critic'])
        for i,name in [(0,'fit'),(2,'validation')]:
            np.testing.assert_allclose(calls[0][i][2],scores[splits['source'][name]],rtol=1e-6)
            self.assertTrue((calls[0][i+1][2]==1).all())
        xs,sl,sy,xt,tl=data()
        critic=torch.nn.Linear(4,2)
        critic.load_state_dict(artifacts['critics']['source_weighted_critic'])
        si,ti=splits['source']['eval'],splits['target']['eval']
        with torch.no_grad():
            expected=dis2_components((sl[si].argmax(1)!=sy[si]).numpy(),
                (critic(xs[si]).argmax(1)!=sl[si].argmax(1)).numpy(),
                (critic(xt[ti]).argmax(1)!=tl[ti].argmax(1)).numpy())
        for key in expected: self.assertAlmostEqual(rows[0][key],expected[key])

    def test_empty_pruning_is_unsupported_not_zero(self):
        rows,_,_=self.run_fixed(np.full(100,.2),methods=['source_pruning'],thresholds=[.9])
        self.assertEqual(rows[0]['status'],'unsupported_source_eval')
        self.assertTrue(np.isnan(rows[0]['lower_bound']))
        self.assertFalse(rows[0]['confidence_bound_available'])

    def test_real_discriminator_cli_target_label_isolation_and_plots(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); folder=root/'features/cifar10/ERM-aug-imagenet_1_100.0'
            folder.mkdir(parents=True)
            xs,sl,sy,xt,tl=data()
            np.save(folder/'model_feats.npy',{'source':xs.numpy(),'target':xt.numpy()})
            labels={'source':sy.numpy(),'target':(xt[:,1]>0).long().numpy()}
            np.save(folder/'ytrue.npy',labels)
            head=torch.nn.Linear(4,2)
            with torch.no_grad():
                head.weight.zero_(); head.weight[0,0]=-1; head.weight[1,0]=1; head.bias.zero_()
            torch.save(head.state_dict(),folder/'linear.pth')
            args=['--feats_dir',str(root/'features'),'--results_dir',str(root/'results'),
                '--datasets','cifar10','--shifts','1','--train_methods','ERM-aug-imagenet',
                '--source_thresholds','0','.99','--epochs','2','--domain_epochs','2',
                '--critic_repeats','2','--include_dis2_reference','--device','cpu',
                '--bound_strategies','logits','features','PCA1','PCA4','PCA16','PCA32','PCA64','PCA128']
            with contextlib.redirect_stdout(io.StringIO()): path=evaluate_cli(args)
            original=pd.read_pickle(path)
            self.assertEqual(len(original),32)
            self.assertEqual(len(list((root/'results').glob('*models*.pth'))),8)
            labels['target']=1-labels['target']; np.save(folder/'ytrue.npy',labels)
            with contextlib.redirect_stdout(io.StringIO()): evaluate_cli(args)
            changed=pd.read_pickle(path)
            for column in ['lower_bound','error_estimate_raw','critic_selection_score']:
                np.testing.assert_allclose(original[column],changed[column],equal_nan=True)
            with contextlib.redirect_stdout(io.StringIO()):
                table=plot_cli(['--results',str(path),'--plot_dir',str(root/'plots'),
                    '--methods','source_weighted_critic','--figures','comparison','representations'])
            self.assertEqual(len(list((root/'plots').glob('*.png'))),3)
            self.assertEqual(len(table),9)
            points=pd.read_csv(next((root/'plots').glob('*points.csv')))
            np.testing.assert_allclose(points[points['plot']=='comparison'].prediction,
                changed[(changed.prediction_method=='source_weighted_critic') & (changed.bound_strategy=='logits')].lower_bound)


if __name__=='__main__': unittest.main()
