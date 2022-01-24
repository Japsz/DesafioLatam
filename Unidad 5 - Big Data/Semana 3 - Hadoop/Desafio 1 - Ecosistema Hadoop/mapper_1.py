#!/usr/bin/python3
import re, sys

line_feed = sys.stdin

for line in line_feed:
	(tag_id, idx, relevance) = re.sub('\n', '', line).split(',')
	print(f'{tag_id},{relevance}')
