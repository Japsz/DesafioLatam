#!/usr/bin/python3
import re, sys

line_feed = sys.stdin

tag_relevance = dict()
for line in line_feed:
	(tag_id, relevance) = re.sub('\n', '', line).split(',')
	if tag_id in tag_relevance.keys():
		tag_relevance[tag_id].append(float(relevance))
	else:
		tag_relevance[tag_id] = [float(relevance)]

for tag_id, relevances in tag_relevance.items():
	print(f'{tag_id}={sum(relevances)/len(relevances)}')
