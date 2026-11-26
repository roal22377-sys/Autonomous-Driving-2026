import os
from pathlib import Path
import unittest
import numpy as np
import torch
import train_il_s2r as IL
from utils.env import F1Wrapper
from utils.training_mix import hall_split, groups_for, family, step_budgets
from utils.policy_transfer import columns, expand_state
from ruamel.yaml import YAML


class HairpinTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = IL.get_args(['--task_config', 'configs/task/f1tenth_preview_1126.yaml'])

    def test_feature_migration_preserves_original_first_layer(self):
        old = YAML().load(open('results/ppo_s2r/2130/backup/f1tenth.yaml'))
        before, after = columns(old), columns(self.args)
        self.assertEqual((len(before), len(after)), (129, 132))
        rng = np.random.default_rng(12)
        old_mean, new_mean = rng.normal(size=129), rng.normal(size=132)
        old_var, new_var = rng.uniform(.1, 2, 129), rng.uniform(.1, 2, 132)
        state = {'fc1.weight': torch.tensor(rng.normal(size=(16,129)), dtype=torch.float32),
                 'fc1.bias': torch.tensor(rng.normal(size=16), dtype=torch.float32)}
        migrated = expand_state(state,before,after,old_mean,old_var,new_mean,new_var)
        raw = rng.normal(size=(40,132))
        idx = [after.index(x) for x in before]
        old_z = torch.tensor((raw[:,idx]-old_mean)/np.sqrt(old_var+1e-8),dtype=torch.float32)
        new_z = torch.tensor((raw-new_mean)/np.sqrt(new_var+1e-8),dtype=torch.float32)
        a = old_z @ state['fc1.weight'].T + state['fc1.bias']
        b = new_z @ migrated['fc1.weight'].T + migrated['fc1.bias']
        torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-5)
        added = [i for i,x in enumerate(after) if x not in before]
        self.assertEqual(len(added),3)
        self.assertEqual(float(migrated['fc1.weight'][:,added].abs().sum()),0.)

    def test_2025_copy_is_trained_but_other_hall_holdouts_are_not(self):
        train,valid = hall_split()
        self.assertEqual(len([p for p in train if Path(p).name.startswith('Course2025')]),2)
        self.assertFalse(any(Path(p).name.startswith('Course2025') for p in valid))
        self.assertFalse(set(train)&set(valid))
        self.assertGreater(len(valid),40)
        regular=[p for p in IL.list_variants(IL.MAPS) if IL.HOLDOUT not in p]
        groups=groups_for(self.args,regular)
        self.assertTrue(all(groups[g] for g in self.args.training_mix))
        self.assertTrue(all(family(p)==g for g,ps in groups.items() for p in ps))

    def test_step_budget_is_exact_and_not_episode_count(self):
        total=100003
        b=step_budgets(self.args.training_mix,total)
        self.assertEqual(sum(b.values()),total)
        for k,v in self.args.training_mix.items():self.assertLess(abs(b[k]/total-v),1/total)

    def test_nominal_lateral_plan_is_six(self):
        path='installation/maps_real/Practice1126b.npz'
        env=F1Wrapper(self.args,IL.MAPS,obs_noise=False,sim_mode='eval',tracks=[path])
        try:
            obs,_=env.reset(options={'track':path})
            self.assertEqual(obs.shape,(132,))
            self.assertTrue(np.isfinite(obs).all())
            self.assertAlmostEqual(IL.expert_kwargs(env)['a_lat'],6.,places=5)
            p=env.preview_cfg['plan']
            self.assertAlmostEqual(p['grip']*p['grip_share'],6.,places=5)
        finally:env.close()

    def test_expert_wheel_angles_round_trip_through_linkage(self):
        path='installation/maps_real/Practice1126b.npz'
        env=F1Wrapper(self.args,IL.MAPS,obs_noise=False,sim_mode='eval',tracks=[path])
        try:
            env.reset(options={'track':path})
            env.expert_cfg=dict(env.expert_cfg,invert_steer_response=True)
            for scale in [.8,.9,1.05,1.15]:
                env.steer_scale=(scale,scale)
                for desired in np.linspace(-.5,.5,41):
                    command=env.steer_command(desired)
                    lock=env.steer_left_max if command>0 else env.steer_right_max
                    physical=command*lock*(scale+(1-scale)*command**2)
                    self.assertAlmostEqual(physical,desired,places=7)
        finally:env.close()

    def test_straight_gate_covers_straights_and_excludes_hairpins(self):
        path='installation/maps_real/Practice1126b.npz'
        env=F1Wrapper(self.args,IL.MAPS,obs_noise=False,sim_mode='eval',tracks=[path])
        try:
            env.reset(options={'track':path})
            line=env._line_ref();state=env.vehicle_state().copy()
            original=env.vehicle_state
            env.vehicle_state=lambda:state
            straight=corner_count=0
            for j in range(line.n):
                state[:2]=line.xy[j];state[4]=line.heading[j]
                env.elapsed+=env.timestep;env._reward_line_hint=j
                flag=env.on_straight();straight+=int(flag)
                if abs(line.k[j])>.5:
                    corner_count+=1;self.assertFalse(flag)
            self.assertGreater(straight/line.n,.05)
            self.assertGreater(corner_count,0)
            env.vehicle_state=original
        finally:env.close()


if __name__=='__main__':unittest.main()
