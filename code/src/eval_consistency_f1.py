from helper import *

grouping = {
	'base': ['0', 'neg0'],
	'op only': ['1', '2', 'neg1', 'neg2'],
	'op + neg': ['3', '4', '5', '6', 'neg3', 'neg4', 'neg5', 'neg6']
}

grouping_neg = {
	'base': ['0'],
	'neg only':['1','2','3'],
}

def compute_metrics_by_theory(results_by_theory, args):
	consistency_micro, consistency_weighted = [], []
	for k,v in results_by_theory.items():
		y_true, y_pred = [x[0] for x in v], [x[1] for x in v]
		if args.type == 'easy':
			consistency_micro.append(np.round(f1_score(y_true, y_pred, average='micro'), 2))
			consistency_weighted.append(np.round(f1_score(y_true, y_pred, average='weighted'), 2))
		elif args.type == 'strict':
			consistency_micro.append(int(y_true == y_pred))
			consistency_weighted.append(int(y_true == y_pred))

	return np.round(np.mean(consistency_micro), 2), np.round(np.mean(consistency_weighted), 2)

def main(args):
	results_by_theory, all_results_summary = ddict(list), ddict(int)
	results_by_version, results_by_label, results_by_version_label = ddict(lambda: ddict(list)), ddict(lambda: ddict(list)), ddict(lambda: ddict(lambda: ddict(list)))
	all_tgts, all_preds = [], []
	path_read = f'../saved/{args.outfile}/iid_output.csv'

	with open(path_read, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			theory_id, tgt, pred = row[0], int(row[1]), int(row[2])
			all_tgts.append(tgt)
			all_preds.append(pred)
			if '@' in theory_id:
				# format is 7dfe482a_equiv_type1@16, 7dfe482a_@16
				part1, part2 = theory_id.split('@')
				part3, version = part1.split('_', 1) if '_' in part1 else (part1, 'base')
				theory = part3 + f'@{part2}'
			else:
				theory, version = theory_id.split('_', 1) if '_' in theory_id else (theory_id, 'base')
			results_by_theory[theory].append((tgt, pred))
			results_by_version[version][theory].append((tgt, pred))
			results_by_label[tgt][theory].append((tgt, pred))
			results_by_version_label[version][tgt][theory].append((tgt, pred))

	consistency_micro, consistency_weighted = compute_metrics_by_theory(results_by_theory, args)

	print('****************************')
	print('Overall results')
	# print(f'{np.round(np.mean(consistency_micro), 2)}')
	print(f'{np.round(np.mean(consistency_weighted), 10)}')
	all_results_summary['overall'] = f'{np.round(np.mean(consistency_weighted), 10)}'

	if args.grouped:
		print('****************************')
		operator = list(results_by_version.keys())[0][:4]
		all_results_summary['operator'] = operator
		if 'neg' in operator:
			operator = 'neg'
			grouping_final = grouping_neg
		else:
			grouping_final = grouping
		print(f'Grouped results weighted for operator {operator}:')
		group_keys, group_res = [], []

		for k, v in grouping_final.items():
			group_outputs = None
			for group_id in v:
				if group_outputs is None:
					group_outputs = results_by_version[f'{operator}{group_id}']
				else:
					for gk, gv in results_by_version[f'{operator}{group_id}'].items():
						group_outputs[gk].extend(gv)


			group_keys.append(f"{k}")
			group_val = f"{compute_metrics_by_theory(group_outputs, args)[1]}"
			group_res.append(group_val)
			all_results_summary[f'{k}'] = group_val

		for x in group_keys:
			print(x)
		print('*****')
		for x in group_res:
			print(x)

	if args.by_label:
		results_lbl_weighted = []
		for k,v in results_by_label.items():
			results_lbl_weighted.append(f"{k},{compute_metrics_by_theory(v, args)[1]}")

		results_lbl_weighted = sorted(results_lbl_weighted)

		print('****************************')
		print('Weighted results by label')
		for item in results_lbl_weighted:
			print(item.split(',')[0])
		print('*****')
		tmp = []
		for item in results_lbl_weighted:
			tmp.append(float(item.split(',')[1]))
		print(tmp)

		for item in results_lbl_weighted:
			part0, part1 = item.split(',')
			all_results_summary[f'label_{part0}'] = float(part1)


	if args.grouped_and_label:

		print('****************************')
		print('Weighted results by version and label')
		results_ver_lbl_weighted = ddict(lambda : ddict(lambda : ddict(list)))
		operator = list(results_by_version.keys())[0][:4]
		if 'neg' in operator:
			operator = 'neg'
			grouping_final = grouping_neg
		else:
			grouping_final = grouping

		group_result_key = []
		for k,v in grouping_final.items():
			for group_id in v:
				group_version = results_by_version_label[f'{operator}{group_id}']
				for label_key,label_value in group_version.items():
					if bool(results_ver_lbl_weighted[k][label_key]):
						for gk, gv in label_value.items():
							results_ver_lbl_weighted[k][label_key][gk].extend(gv)
					else:
						results_ver_lbl_weighted[k][label_key] = label_value

		for version_key,version_value in results_ver_lbl_weighted.items():
			for label_key,label_value in version_value.items():
				print(f'This is version {version_key} and label {label_key}, its acc is {compute_metrics_by_theory(label_value,args)[1]}')

	return dict(all_results_summary)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Evaluate Consistency Metric')
	parser.add_argument('--outfile')
	parser.add_argument('--grouped', action='store_true')
	parser.add_argument('--by_label', action='store_true')
	parser.add_argument('--grouped_and_label', action='store_true')
	parser.add_argument('--type', default='easy', choices=['strict', 'easy'])
	parser.add_argument('--filter', default='none')
	parser.add_argument('--model', default='t5_large',choices=['t5_large','roberta','gpt3'])
	args = parser.parse_args()

	main(args)
