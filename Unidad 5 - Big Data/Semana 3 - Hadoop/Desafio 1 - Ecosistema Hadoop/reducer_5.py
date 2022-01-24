#!/usr/bin/python3
import re, sys

line_feed = sys.stdin

movie_years = dict()
for line in line_feed:
	(movie_id, year) = re.sub('\n', '', line).split(',')
	year = int(year)
	if year in movie_years.keys():
		movie_years[year] += 1
	else:
		movie_years[year] = 1

for year, cant in movie_years.items():
	print(f'{year}={cant}')

