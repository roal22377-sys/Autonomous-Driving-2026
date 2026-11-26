import tempfile
from pathlib import Path
import unittest
import numpy as np
import train_il_s2r as IL
from utils.policy_transfer import columns
from utils.replay import load_replay
from utils.vectorize import RunningMeanStd
from ruamel.yaml import YAML


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.args=IL.get_args(['--task_config','configs/task/f1tenth_preview_1126.yaml','--batch','32'])
        self.args.replay_file=str(Path(self.tmp.name)/'cache.npz')
        self.rms=RunningMeanStd('obs',132);self.rms.mean[:]=1.;self.rms.var[:]=4.
        old=YAML().load(Path('results/ppo_s2r/2130/backup/f1tenth.yaml'))
        self.names=columns(self.args);self.old_names=columns(old)
        self.data=dict(O=np.zeros((8,132),np.float16),Y=np.tile([.2,-.3],(8,1)).astype(np.float32),
                       columns=np.asarray(self.names),old_columns=np.asarray(self.old_names),
                       missing_new_features=np.array([True,False]*4),teacher=np.asarray('ppo_s2r/2130'))
        self.data['O'][:,self.names.index('preview/valid')]=1.

    def tearDown(self):self.tmp.cleanup()

    def load(self):
        np.savez(self.args.replay_file,**self.data)
        return load_replay(self.args,self.rms)

    def test_unknown_features_have_no_normalized_contribution(self):
        replay=self.load();new=[i for i,k in enumerate(self.names) if k not in self.old_names]
        np.testing.assert_array_equal(replay.X[0,new].numpy(),0.)
        np.testing.assert_allclose(replay.X[1,new].numpy(),-.5,atol=1e-4)
        self.assertEqual(replay.batch,16)
        self.assertEqual(len(replay.train)+len(replay.val),8)

    def test_reordered_inputs_and_changed_action_scaling_are_rejected(self):
        self.data['columns'][[0,1]]=self.data['columns'][[1,0]]
        with self.assertRaisesRegex(ValueError,'schema'):self.load()
        self.data['columns']=np.asarray(self.names);self.args.max_speed=6.
        with self.assertRaisesRegex(ValueError,'action scaling'):self.load()

    def test_invalid_localization_implies_raw_zero_preview_not_neutral_input(self):
        self.data['O'][0,self.names.index('preview/valid')]=0.
        replay=self.load();new=[i for i,k in enumerate(self.names) if k not in self.old_names]
        np.testing.assert_allclose(replay.X[0,new].numpy(),-.5,atol=1e-4)
        np.testing.assert_array_equal(replay.X[2,new].numpy(),0.)

    def test_invalid_labels_or_no_new_experience_are_rejected(self):
        self.data['Y'][0,0]=np.nan
        with self.assertRaises(ValueError):self.load()
        self.data['Y'][0,0]=.2;self.args.replay_share=1.
        with self.assertRaisesRegex(ValueError,'both new and replay'):self.load()


if __name__=='__main__':unittest.main()
