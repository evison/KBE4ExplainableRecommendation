import os,sys
from math import log

rank_list_file = sys.argv[1]
test_qrel_file = sys.argv[2]
rank_cutoff_list = [int(cut) for cut in sys.argv[3].split(',')]

#read qrel file
qrel_map = {}
with open(test_qrel_file) as fin:
	for line in fin:
		arr = line.strip().split(' ')
		qid = arr[0]
		did = arr[2]
		label = int(arr[3])
		if label < 1:
			continue
		if qid not in qrel_map:
			qrel_map[qid] = set()
		qrel_map[qid].add(did)

#compute ndcg
def metrics(doc_list, rel_set):
	dcg = 0.0
	hit_num = 0.0
	for i in xrange(len(doc_list)):
		if doc_list[i] in rel_set:
			#dcg
			dcg += 1/(log(i+2)/log(2))
			hit_num += 1
	#idcg
	idcg = 0.0
	for i in xrange(min(len(rel_set),len(doc_list))):
		idcg += 1/(log(i+2)/log(2))
	ndcg = dcg/idcg
	recall = hit_num / len(rel_set)
	precision = hit_num / len(doc_list)
	#compute hit_ratio
	hit = 1.0 if hit_num > 0 else 0.0
	large_rel = 1.0 if len(rel_set) > len(doc_list) else 0.0
	return recall, ndcg, hit, large_rel, precision

def print_metrics_with_rank_cutoff(rank_cutoff):
	#read rank_list file
	rank_list = {}
	with open(rank_list_file) as fin:
		for line in fin:
			arr = line.strip().split(' ')
			qid = arr[0]
			did = arr[2]
			if qid not in rank_list:
				rank_list[qid] = []
			if len(rank_list[qid]) > rank_cutoff:
				continue
			rank_list[qid].append(did)

	ndcgs = 0.0
	recalls = 0.0
	hits = 0.0
	large_rels = 0.0
	precisions = 0.0
	count_query = 0
	for qid in rank_list:
		if qid in qrel_map:
			recall, ndcg, hit, large_rel, precision = metrics(rank_list[qid],qrel_map[qid])
			count_query += 1
			ndcgs += ndcg
			recalls += recall
			hits += hit
			large_rels += large_rel
			precisions += precision

	print("Query Number:" + str(count_query))
	print("Larger_rel_set@"+str(rank_cutoff) + ":" + str(large_rels/count_query))
	print("Recall@"+str(rank_cutoff) + ":" + str(recalls/count_query))
	print("Precision@"+str(rank_cutoff) + ":" + str(precisions/count_query))
	print("NDCG@"+str(rank_cutoff) + ":" + str(ndcgs/count_query))
	print("Hit@"+str(rank_cutoff) + ":" + str(hits/count_query))

for rank_cut in rank_cutoff_list:
	print_metrics_with_rank_cutoff(rank_cut)
