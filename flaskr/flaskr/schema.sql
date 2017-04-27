
drop table if exists entries;
create table entries (
  id integer primary key autoincrement,
  age integer not null,
  gender integer not null,
  Cp integer NOT NULL,               
  Trestbps integer NOT NULL,
  Chol integer NOT NULL,
  Fbs integer NOT NULL,
  Restecg integer NOT NULL,
  Thalach integer NOT NULL,
  Exang integer NOT NULL,
  Old_Peak_ST numeric(5,2) NOT NULL,
  Slope integer NOT NULL,
  Ca integer NOT NULL,
  Thal integer NOT NULL  

);

