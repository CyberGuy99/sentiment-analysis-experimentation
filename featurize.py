# This function generates feature vector in sparse format for every sentence pair 
def featurize(index,unigram,bigram):
# Presence not Counts
	for itm in unigram:
		try:
			col_ind1 = lookup[itm]
			#col_ind2 = col_ind1+1
			row.append(index)
			#row.append(index)
			col.append(col_ind1)
			#col.append(col_ind2)
			#data.append(1)
			#data.append(1)
		except Exception:
			continue			
		
	for itm in bigram:
		try:
			col_ind1 = lookup[itm]
			#col_ind2 = col_ind1+1
			row.append(index)
			#row.append(index)
			col.append(col_ind1)
			#col.append(col_ind2)
			#data.append(1)
			#data.append(1)
		except Exception:
			continue
	return None