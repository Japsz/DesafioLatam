#!/usr/bin/python3
import re, sys

line_feed = sys.stdin

for line in line_feed:
	(user_id, movie_id, rating, time) = re.sub('\n', '', line).split(',')
	print(f'{movie_id},{rating}')
