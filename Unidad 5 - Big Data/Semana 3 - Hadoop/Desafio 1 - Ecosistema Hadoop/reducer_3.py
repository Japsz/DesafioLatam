#!/usr/bin/python3
import re, sys

line_feed = sys.stdin

movie_ratings = dict()
for line in line_feed:
	(movie_id, rating) = re.sub('\n', '', line).split(',')
	if movie_id in movie_ratings.keys():
		movie_ratings[movie_id].append(float(rating))
	else:
		movie_ratings[movie_id] = [float(rating)]

for movie_id, ratings in movie_ratings.items():
	print(f'{movie_id},{sum(ratings)/len(ratings)}')
