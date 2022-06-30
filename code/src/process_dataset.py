from helper import *


csv.field_size_limit(sys.maxsize)

total_instances, truncated_instances = 0, 0

class LogicalInstance:

	def __init__(self, context, statement, label, proofs, proof_deps, has_neg, theory_id):
		self.context     = context
		self.statement   = statement
		self.label       = label
		self.proofs      = proofs
		self.proof_deps  = proof_deps
		self.has_neg     = has_neg
		self.theory_id   = theory_id

	@classmethod
	def from_csv(cls, row):
		proof_deps = row[4]
		proof_deps = list(map(int, proof_deps.split(',')))

		if len(row) == 7:
			return LogicalInstance(row[0], row[1], int(row[2]), row[3], proof_deps, row[5], row[6])
		else:
			return LogicalInstance(row[0], row[1], int(row[2]), row[3], proof_deps, row[5], '')

	def no_tokenize(self,tokenizer):
		# if tokenizer is not None:
		# 	sentence = "Theory: " + self.context + " Statement: " + self.statement
		# 	count = len(tokenizer(sentence)['input_ids'])
		# 	return sentence,count

		input_str = '$answer$ ; $question$ = '
		self.statement = self.statement[:-1] + '?'
		input_str += self.statement + ' ; $context$ ='
		input_str_split = input_str
		ctx = ''
		for id, sent in enumerate(self.context.split('. ')):
			sent = sent + '.'
			ctx += f' {sent}'
		input_str_split += ctx
		input_str_split = input_str_split[:-1]
		count = len(tokenizer(input_str_split)['input_ids'])
		return input_str_split,count


	def tokenize_ptlm(self, tokenizer, filtered, ques_only=False):
		# convert the data in the format expected by the PTLM
		# format: [CLS] context [SEP] statement [SEP]

		global total_instances, truncated_instances

		if ques_only:
			input_tokens = tokenizer.cls_token + self.statement + tokenizer.sep_token
		else:
			input_tokens = tokenizer.cls_token + self.context + tokenizer.sep_token + self.statement + tokenizer.sep_token

		input_ids    = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_tokens))
		ctx_ids      = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tokenizer.cls_token + self.context + tokenizer.sep_token))
		stmt_ids     = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tokenizer.cls_token + self.statement + tokenizer.sep_token))

		total_instances += 1

		if len(input_ids) > tokenizer.model_max_length:
			if filtered:
				return None
			else:
				truncated_instances += 1

		return input_ids[:tokenizer.model_max_length], ctx_ids[:tokenizer.model_max_length], stmt_ids

	def tokenize_ptlm_t5(self, tokenizer, filtered, ques_only=False):
		global total_instances, truncated_instances

		input_str = '$answer$ ; $question$ = '
		self.statement = self.statement[:-1] + '?'
		input_str += self.statement + ' ; $context$ ='
		input_str_split = input_str
		ctx = ''
		for id,sent in enumerate(self.context.split('. ')):
			sent = sent + '.'
			# ctx += f' sent{id+1}: {sent}'
			ctx += f' {sent}'
		input_str_split += ctx
		input_str_split = input_str_split[:-1]


		input_str_nltk = input_str
		ctx = ''
		for id,sent in enumerate(sent_tokenize(self.context)):
			# ctx += f' sent{id+1}: {sent}'
			ctx += f' {sent}'
		input_str_nltk += ctx

		assert input_str_split == input_str_nltk
		input_str = input_str_nltk
		input_str += tokenizer.eos_token
		input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))

		# input_str = '$answer$ ; $question$ = What is one single-hop inference? ; $context$ ='

		label = map_int_to_str(self.label)
		output_ids = tokenizer(f'$answer$ = {label}').input_ids

		total_instances += 1

		if len(input_ids) > tokenizer.model_max_length:
			if filtered:
				return None
			else:
				truncated_instances += 1


		return input_ids[:tokenizer.model_max_length], output_ids[:tokenizer.model_max_length]


	def tokenize(self, tokenizer, arch, split, filtered, ques_only=False):
		if arch == 'roberta_large_race':
			return self.tokenize_ptlm(tokenizer, filtered, ques_only=ques_only)
		if arch == 'albert_xxlarge_v2':
			return self.tokenize_ptlm(tokenizer, filtered, ques_only=ques_only)
		if arch == 't5_large':
			return self.tokenize_ptlm_t5(tokenizer, filtered, ques_only=ques_only)



def map_int_to_str(label):
	if label == 0:
		label = 'False'
	elif label == 1:
		label = 'True'
	elif label == 2:
		label = 'Nothing'
	return label

def get_inp_fname(args, split):
	if args.equiv:
		return f'../data/{args.dataset}/{args.dataset_type}/{split}_equiv.csv'
	if args.equiv2:
		return f'../data/{args.dataset}/{args.dataset_type}/{split}_equiv2.csv'
	else:
		return f'../data/{args.dataset}/{args.dataset_type}/{split}.csv'

def get_out_fname(args, split, key=None, dir_name=False):
	dataset = f'{args.dataset}_{args.dataset_type}'
	if args.equiv:
		dataset = f'{dataset}_equiv'
	if args.equiv2:
		dataset = f'{dataset}_equiv2'
	if args.filtered:
		dataset = f'{dataset}_filtered'
	if args.ques_only:
		dataset = f'{dataset}_quesonly'

	if dir_name:
		return f'../data/processed/{dataset}/{args.arch}/{split}/'
	else:
		if args.arch.startswith('csv_file'):
			return f'../data/processed/{dataset}/{args.arch}/{split}/{split}.csv'
		else:
			return f'../data/processed/{dataset}/{args.arch}/{split}/{key}.pkl'


def main(args):

	assert args.filtered, "Please use --filtered to ensure max token lengths are not exceeded!"
	assert args.trim, "Please use --trim to get valid train/dev/test lengths!"

	if args.trim:
		if args.eval:
			test_length = 20000
		else:
			train_length = 50000
			dev_length   = 10000
			test_length  = 10000

	if os.path.exists(get_out_fname(args, 'test', key='has_neg')) and not args.force:
		print(f"Dataset {get_out_fname(args, 'test', dir_name=True)} exists! Skipping!!")
		sys.exit(0)

	if args.force:
		print(f"Dataset {get_out_fname(args, 'test', dir_name=True)} exists! Overwriting now!!")

	# load tokenizer
	if args.arch == 'roberta_large_race':
		tokenizer = AutoTokenizer.from_pretrained("LIAMF-USP/roberta-large-finetuned-race")
	elif args.arch == 'albert_xxlarge_v2':
		tokenizer = AutoTokenizer.from_pretrained('albert-xxlarge-v2')

	elif args.arch == 't5_large':
		tokenizer = AutoTokenizer.from_pretrained('t5-large')

	elif args.arch.startswith('csv_file'):
		if "gpt3" in args.arch:
			tokenizer = AutoTokenizer.from_pretrained('gpt2')
		elif "t5_11b" in args.arch:
			tokenizer = AutoTokenizer.from_pretrained('t5-large')

		print('Would not use tokenizer but only use tokenizer to count the # tokens and output csv file')
	else:
		print('Token type ids not implemented in tokenize call, will not work for bert models')
		import pdb; pdb.set_trace()
		raise NotImplementedError

	# load data
	for split in ['train', 'dev', 'test']:

		print(f'Processing {split} split...')

		# make folder if not exists
		print(f'Creating directory {get_out_fname(args, split, dir_name=True)}')
		pathlib.Path(get_out_fname(args, split, dir_name=True)).mkdir(exist_ok=True, parents=True)

		data, metadata = ddict(list), dict()



		if args.arch.startswith('csv_file'):
			with open(get_inp_fname(args, split)) as f:
				count_all = 0
				reader = csv.reader(f)
				for row in tqdm(reader):
					instance = LogicalInstance.from_csv(row)
					output,count_prompt = instance.no_tokenize(tokenizer)

					if output is not None:
						completion = f'$answer$ = {map_int_to_str(instance.label)}'
						count_completion = len(tokenizer(completion)['input_ids'])
						if (count_completion + count_prompt) < tokenizer.model_max_length and args.filtered:
							data['prompt'].append(output)
							data['completion'].append(completion)
						elif not args.filtered:
							data['prompt'].append(output)
							data['completion'].append(completion)

					count_all += (count_prompt+count_completion)

			data = dict(data)
			if args.trim:
				if split == 'test':
					data['prompt'] = data['prompt'][:test_length]
					data['completion'] = data['completion'][:test_length]
				elif split == 'train' and not args.eval:
					data['prompt'] = data['prompt'][:train_length]
					data['completion'] = data['completion'][:train_length]

				elif split == 'dev' and not args.eval:
					data['prompt'] = data['prompt'][:dev_length]
					data['completion'] = data['completion'][:dev_length]
			print(f"After trim and filter, the final size of {split} data is {len(data['completion'])}")

			data = pd.DataFrame.from_dict(data)
			data.to_csv(get_out_fname(args, split),index = False)
			if count_all > 0:
				print(f'Total number of tokens for tokenizer is {count_all} for {split} data')

		elif args.arch == 'roberta_large_race' or args.arch == 'albert_xxlarge_v2':
			# load the relevant problog file and select all the data
			with open(get_inp_fname(args, split)) as f:
				reader = csv.reader(f)
				for row in tqdm(reader):
				# for idx, row in enumerate(reader):
					# print(idx)
					instance = LogicalInstance.from_csv(row)
					output   = instance.tokenize(tokenizer, args.arch, split, args.filtered, args.ques_only)
					if output is not None:
						data['input_ids'].append(output[0])
						data['ctx_ids'].append(output[1])
						data['stmt_ids'].append(output[2])
						data['label'].append(instance.label)
						data['deps'].append(instance.proof_deps)
						data['has_neg'].append(instance.has_neg)
						data['theory_id'].append(instance.theory_id)

			data = dict(data)

			# write the data in pickle format to processed folder
			for key in ['input_ids', 'ctx_ids', 'stmt_ids', 'label', 'deps', 'has_neg', 'theory_id']:
				print(f'Contains {len(data[key])} lines for dataset: {args.dataset} split: {split} key: {key}')
				if args.trim:
					if split == 'test':
						data[key] = data[key][:test_length]
					elif split == 'train' and not args.eval:
						data[key] = data[key][:train_length]
					elif split == 'dev' and not args.eval:
						data[key] = data[key][:dev_length]
				print(f'Dumping {len(data[key])} lines for dataset: {args.dataset} split: {split} key: {key}')
				with open(get_out_fname(args, split, key=key, dir_name=False), 'wb') as f:
					pickle.dump(data[key], f)

			print(f'Percent truncated: {100 * truncated_instances / total_instances}')

		elif args.arch == 't5_large':
			# load the relevant problog file and select all the data
			with open(get_inp_fname(args, split)) as f:
				reader = csv.reader(f)
				for row in tqdm(reader):
					# for idx, row in enumerate(reader):
					# print(idx)
					instance = LogicalInstance.from_csv(row)
					output = instance.tokenize(tokenizer, args.arch, split, args.filtered, args.ques_only)
					if output is not None:
						data['input_ids'].append(output[0])
						data['output_ids'].append(output[1])
						data['has_neg'].append(instance.has_neg)
						data['theory_id'].append(instance.theory_id)

			data = dict(data)

			# write the data in pickle format to processed folder
			for key in ['input_ids', 'output_ids','has_neg', 'theory_id']:
				print(f'Contains {len(data[key])} lines for dataset: {args.dataset} split: {split} key: {key}')
				if args.trim:
					if split == 'test':
						data[key] = data[key][:test_length]
					elif split == 'train' and not args.eval:
						data[key] = data[key][:train_length]
					elif split == 'dev' and not args.eval:
						data[key] = data[key][:dev_length]

				print(f'Dumping {len(data[key])} lines for dataset: {args.dataset} split: {split} key: {key}')
				with open(get_out_fname(args, split, key=key, dir_name=False), 'wb') as f:
					pickle.dump(data[key], f)

			print(f'Percent truncated: {100 * truncated_instances / total_instances}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Preprocess data')

	parser.add_argument('--dataset')
	parser.add_argument('--dataset_type')
	parser.add_argument('--filtered', action='store_true')
	parser.add_argument('--trim', action='store_true')
	parser.add_argument('--eval', action='store_true')
	parser.add_argument('--equiv', action='store_true')
	parser.add_argument('--equiv2', action='store_true')
	parser.add_argument('--force', action='store_true')
	parser.add_argument('--ques_only', action='store_true')
	parser.add_argument('--arch', default='roberta_large_race', choices=['roberta_large_race','albert_xxlarge_v2','t5_large','csv_file_gpt3',"csv_file_t5_11b"])

	args = parser.parse_args()

	main(args)
