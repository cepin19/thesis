nohup python -u read_natur2.py --output_dir=newtest_skip_sent --model gpt2 --skip-first=True --sent-seg=True   > newtest_skip_sent.log &
nohup python -u read_natur2.py --output_dir=newtest_skip --model gpt2 --skip-first=True --sent-seg=False   > newtest_skip.log &
nohup python -u read_natur2.py --output_dir=newtest --model gpt2 --skip-first=False --sent-seg=False   > newtest.log &
nohup python -u read_natur2.py --output_dir=newtest_sent --model gpt2 --skip-first=False --sent-seg=True   > newtest_sent.log &
