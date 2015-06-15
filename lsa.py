def lsa(D, k):
	counter = 0
	# Phase 1-initialization
	for t in D:
		counter += 1
		if counter <= k:
			#label t as an outlier with flag "1
		else:
			#update hash tables using t
			# label t as a non-outlier with flag "0
	
	
	not_moved = True
	while not_moved:
		for d in D:
			if getLabel(d) == 0:
				#foreach record o in current k outliers:
				#	exchanging the label of t with that of o and evaluating the change of entropy
				#if maximal decrease on entropy is achieved by record b:
				#	swap the labels of t and b
				#	update hash tables using t and b
				# 	not_moved = false
				
	#return outliers						 	