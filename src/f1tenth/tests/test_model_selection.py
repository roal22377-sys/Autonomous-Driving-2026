import copy
import unittest
from utils.model_selection import choose_families


class FamilySelectionTests(unittest.TestCase):
    shares=dict(hall=.4,short=.3,regular=.15,open=.05,real=.1)

    def baseline(self):
        return {g:dict(success=1.,res=[(f'{g}{i}@8/center',True,100.) for i in range(4)])
                for g in self.shares}

    def test_hairpin_gain_counts_when_general_pace_is_unchanged(self):
        old=self.baseline();now=copy.deepcopy(old)
        now['hall']['res']=[(key,ok,90.) for key,ok,_ in now['hall']['res']]
        decision=choose_families(now,old,self.shares)
        self.assertTrue(decision['selected'])
        self.assertGreater(decision['relative_pace_gain'],.04)

    def test_faster_general_cannot_hide_narrow_course_failure(self):
        old=self.baseline();now=copy.deepcopy(old)
        now['regular']['res']=[(key,ok,50.) for key,ok,_ in now['regular']['res']]
        now['hall']['success']=.75
        now['hall']['res'][0]=('hall0@8/center',False,1.)
        self.assertFalse(choose_families(now,old,self.shares)['selected'])

    def test_completion_improvement_precedes_time(self):
        now=self.baseline();old=copy.deepcopy(now)
        old['hall']['success']=.75;old['hall']['res'][0]=('hall0@8/center',False,1.)
        now['hall']['res']=[(key,ok,105.) for key,ok,_ in now['hall']['res']]
        self.assertTrue(choose_families(now,old,self.shares)['selected'])

    def test_known_real_time_cannot_select_a_model(self):
        old=self.baseline();now=copy.deepcopy(old)
        now['real']['res']=[(key,ok,50.) for key,ok,_ in now['real']['res']]
        self.assertFalse(choose_families(now,old,self.shares)['selected'])

    def test_known_real_completion_is_still_required(self):
        old=self.baseline();now=copy.deepcopy(old)
        now['hall']['res']=[(key,ok,90.) for key,ok,_ in now['hall']['res']]
        now['real']['success']=.75;now['real']['res'][0]=('real0@8/center',False,1.)
        self.assertFalse(choose_families(now,old,self.shares)['selected'])

    def test_known_completion_improves_without_changing_unseen_score(self):
        now=self.baseline();old=copy.deepcopy(now)
        old['real']['success']=.75;old['real']['res'][0]=('real0@8/center',False,1.)
        decision=choose_families(now,old,self.shares)
        self.assertTrue(decision['selected'])
        self.assertEqual(decision['completion'],decision['reference_completion'])

    def test_changed_cases_or_incorrect_counts_are_rejected(self):
        old=self.baseline();now=copy.deepcopy(old)
        now['hall']['res'][0]=('another@8/center',True,100.)
        with self.assertRaises(ValueError):choose_families(now,old,self.shares)
        now=self.baseline();now['hall']['success']=.75
        with self.assertRaises(ValueError):choose_families(now,old,self.shares)


if __name__=='__main__':unittest.main()
