#!/usr/bin/python3
import re, sys, csv

line_feed = sys.stdin

for line in line_feed:
	(movie_id, name, genres) = list(csv.reader([re.sub('\n', '', line)]))[0]
	print(f'{movie_id},{len(genres.split("|"))}')
