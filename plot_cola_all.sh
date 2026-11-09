shopt -s extglob

for c in cola cola2
do
	for m in SL slor normlp const
	do
	for s in lr_test mlp_test svm_test pearson mlp_scaled_test lme_fn:len_len_slor_normlp
	do
		
	for drop in True False
	do
	python plot.py $m $s plots/"$c"_"$m"_"$s"_drop="$drop".png cola4/"$c"_"$m"*drop=$drop*csv
	python plot.py "$m"_times_len $s plots/"$c"_"$m"_times_len_"$s"_drop="$drop".png cola4/"$c"_"$m"*drop=$drop*csv
done
done
done
done
