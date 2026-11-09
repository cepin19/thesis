shopt -s extglob

for c in cola 
do
	for m in log_prob slor normlp slorr normlpp
	do
	for s in  mlp_test svm_test pearson mlp_scaled_test #lme_fn:len_len_slor_normlp
	do
	for drop in True False
	do
	python plot.py $m $s plots_ntb/"$c"_"$m"_"$s".png  ~/revisiting-uid/src/out/"$c"_"$m"*csv
	python plot.py "$m"_times_len $s plots_ntb/"$c"_"$m"_times_len_"$s".png ~/revisiting-uid/src/out/"$c"_"$m"*csv
done
done
done
done
