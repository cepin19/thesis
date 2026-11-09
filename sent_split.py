from trankit import Pipeline
import sys

p = Pipeline(lang='english', gpu=True, cache_dir='./cache')
i=sys.stdin.read()
sents = p.ssplit(i)
#print(sents)
for s in sents['sentences']:
    print(s['text'])
#    print(s['text'])
#print(sents)
