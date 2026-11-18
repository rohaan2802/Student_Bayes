"""Behavioral checks for data validation, leakage and portable prediction parity."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from analyze import (ROOT,FEATURES,CLASSES,load_data,validate_data,analyze,evaluate,
                     make_classifier,split_rows,portable_model,predict_parameters,wilson_interval)


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame=load_data(ROOT/'data/Syn_student.csv')

    def test_repeatable_split_and_probabilities(self):
        first,pred=analyze(self.frame);second,repeated=analyze(self.frame)
        self.assertEqual(first,second)
        self.assertTrue(pred.equals(repeated))
        self.assertEqual(len(pred),30)
        self.assertEqual(sorted(pred.actual.value_counts()),[10,10,10])
        self.assertTrue(np.allclose(pred[CLASSES].sum(axis=1),1))
        self.assertEqual(len(set(pred.source_row)),30)
        train,test=split_rows(self.frame)
        self.assertFalse(set(train)&set(test))
        self.assertEqual(set(pred.source_row),set(test))

    def test_holdout_changes_cannot_alter_fitted_scaler_or_coefficients(self):
        train,test=split_rows(self.frame)
        exported=portable_model(self.frame)
        changed=self.frame.copy()
        changed.loc[test,FEATURES]+=100
        self.assertEqual(exported,portable_model(changed))
        self.assertTrue(np.allclose(exported['mean'],self.frame.iloc[train][FEATURES].mean()))
        self.assertFalse(np.allclose(exported['mean'],self.frame[FEATURES].mean()))

    def test_portable_model_matches_sklearn_on_every_test_row(self):
        train,test=split_rows(self.frame)
        pipeline=make_classifier().fit(self.frame.iloc[train][FEATURES],self.frame.iloc[train].student_type)
        exported=portable_model(self.frame)
        expected=pipeline.predict_proba(self.frame.iloc[test][FEATURES])
        actual=np.array([[predict_parameters(exported,row)[label] for label in pipeline.classes_]
                         for row in self.frame.iloc[test][FEATURES].to_numpy()])
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)

    def test_training_cv_and_importance_ignore_holdout_values(self):
        _,test=split_rows(self.frame)
        changed=self.frame.copy();changed.loc[test,FEATURES]+=40;changed.loc[test,'quiz_score']+=40
        first,importance=evaluate(self.frame);second,other=evaluate(changed)
        self.assertEqual(first['cv_scores'],second['cv_scores'])
        self.assertTrue(importance.equals(other))
        self.assertNotEqual(first['regression_holdout'],second['regression_holdout'])

    def test_invalid_data_is_rejected(self):
        variants=[]
        variants.append(self.frame.rename(columns={'quiz_score':'score'}))
        missing=self.frame.copy();missing.loc[0,'study_hours']=np.nan;variants.append(missing)
        infinite=self.frame.copy();infinite.loc[0,'study_hours']=np.inf;variants.append(infinite)
        negative=self.frame.copy();negative.loc[0,'study_hours']=-1;variants.append(negative)
        constant=self.frame.copy();constant['study_hours']=2;variants.append(constant)
        wrong_class=self.frame.copy();wrong_class.loc[0,'student_type']='unknown';variants.append(wrong_class)
        variants.append(self.frame.iloc[:10])
        for frame in variants:
            with self.subTest(columns=list(frame.columns),rows=len(frame)):
                with self.assertRaises(ValueError):validate_data(frame)

    def test_small_test_set_has_nonzero_interval_width_even_when_perfect(self):
        low,high=wilson_interval(30,30)
        self.assertAlmostEqual(high,1)
        self.assertGreater(low,.88);self.assertLess(low,.89)
        self.assertEqual(wilson_interval(0,30)[0],0)
        with self.assertRaises(ValueError):wilson_interval(1,0)

    def test_prediction_rejects_bad_values(self):
        model=portable_model(self.frame)
        for values in [[1,2],[-1,2,3],[np.inf,2,3],[np.nan,2,3]]:
            with self.assertRaises(ValueError):predict_parameters(model,values)


if __name__=='__main__':unittest.main()
