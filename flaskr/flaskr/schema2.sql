

drop table if exists patient;
create table patient (
  id2 integer primary key autoincrement,
  SEX integer not null,
  TOTCHOL integer not null,
  AGE integer NOT NULL,
  SYSBP numeric(4,1) NOT NULL,
  DIABP numeric(3,1) NOT NULL,
  CURSMOKE integer NOT NULL,
  CIGPDAY integer NOT NULL,
  BMI numeric(5,2) NOT NULL,
  DIABETES integer NOT NULL,
  EDUC integer NOT NULL,
  HEARTRTE integer NOT NULL,
  GLUCOSE integer NOT NULL,
  BPMEDS integer NOT NULL


);