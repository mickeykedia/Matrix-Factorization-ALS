CREATE TABLE Ratings (
	user_id integer,
	movie_id integer,
	rating float
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3n://netflix-dataset'

CREATE TABLE User_features (
	feature1 float,
	feature2 float,
	feature3 float
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3n://netflix-dataset'

CREATE TABLE Movie_features (
	feature1 float,
	feature2 float,
	feature3 float
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3n://netflix-dataset'