#!/usr/bin/python3
import re, sys, csv

line_feed = sys.stdin

for line in line_feed:
	(movie_id, name, genres) = list(csv.reader([re.sub('\n', '', line)]))[0]
	year = re.findall(r'\(\d{4}\)$', name)
	if year:
		print(f'{movie_id},{re.sub(r"[()]", "", year[0])}')
