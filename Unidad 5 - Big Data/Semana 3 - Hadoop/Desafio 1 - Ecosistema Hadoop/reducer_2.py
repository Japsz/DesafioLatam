#!/usr/bin/python3
import re, sys

line_feed = sys.stdin

user_ratings = dict()
for line in line_feed:
	(user_id, rating) = re.sub('\n', '', line).split(',')
	if user_id in user_ratings.keys():
		user_ratings[user_id].append(float(rating))
	else:
		user_ratings[user_id] = [float(rating)]

for user_id, ratings in user_ratings.items():
	print(f'{user_id}\n\tcant={len(ratings)}\n\tprom={sum(ratings)/len(ratings)}')
