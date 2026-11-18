# Data dictionary

`data/Syn_student.csv` is the unchanged supplied synthetic dataset: 150 rows and five columns. Each category has 50 records. There are no missing values or exact duplicate rows.

| Field | Meaning | Use |
|---|---|---|
| study_hours | Recorded study hours | Predictor; grouped outcome |
| sleep_hours | Recorded sleep hours | Predictor |
| social_time | Recorded social time | Predictor |
| quiz_score | Recorded continuous quiz score | Regression outcome |
| student_type | bad, good, excellent | Classification target; grouping variable |

The sampling frame, generation process, time window for hours, and formal score bounds were not supplied. Category labels are retained verbatim from the course data. They are synthetic labels, not judgments about people. Do not use this demonstration to make decisions about real students.
