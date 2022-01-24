#!/usr/bin/python3
import re, sys

line_feed = sys.stdin

movie_genres = dict()
for line in line_feed:
	(movie_id, genres) = re.sub('\n', '', line).split(',')
	genres = int(genres)
	if genres in movie_genres.keys():
		movie_genres[genres] += 1
	else:
		movie_genres[genres] = 1

for genres, cant in movie_genres.items():
	if genres > 1 and genres < 11:
		print(f'{genres}={cant}')

