import random, json, re, pdb, csv, logging, sys, os, shutil, struct, argparse, math, uuid, jsonlines, types, pathlib, getpass, nltk, itertools, traceback, subprocess, fcntl
import time, socket, itertools
import numpy as np
import networkx as nx
import pickle5 as pickle

import pandas as pd
import wandb

from tqdm import tqdm, trange
from copy import deepcopy
from collections import Counter, OrderedDict
from collections import defaultdict as ddict
from string import Template
from functools import reduce, lru_cache
from operator import mul
from itertools import product
from pprint import pprint
# from pattern import en
from argparse import ArgumentParser
from datetime import datetime, timedelta
from typing import Optional
from typing import Dict, List, NamedTuple, Optional
from configparser import ConfigParser
from pprint import pformat
from ruamel.yaml import YAML
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.nn.init import xavier_normal_, kaiming_uniform_, xavier_uniform_
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.loggers import WandbLogger
# from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from sklearn.metrics import f1_score, confusion_matrix

from nltk.tokenize import word_tokenize, sent_tokenize

import datasets
# from deepspeed.ops.adam import FusedAdam
# from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap

from transformers import (
	AdamW,
	Adafactor,
	AutoModelForSequenceClassification,
	AutoModelForMultipleChoice,
	T5ForConditionalGeneration,
	RobertaModel,
	AutoModel,
	AutoModelWithLMHead,
	AutoConfig,
	AutoTokenizer,
	T5Tokenizer,
	get_scheduler,
	get_linear_schedule_with_warmup,
	glue_compute_metrics
)

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize

random.seed(42)

# setup the logging
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.ERROR)


def freeze_net(module):
	for p in module.parameters():
		p.requires_grad = False

def unfreeze_net(module):
	for p in module.parameters():
		p.requires_grad = True

def clear_cache():
	torch.cuda.empty_cache()

def get_username():
	return getpass.getuser()

def replace_dict(string, dictionary):
	# modified from https://stackoverflow.com/a/6117124/4985078

	dictionary = {f'${{{k}}}': f'${{{v}}}' for k, v in dictionary.items()}
	dictionary = dict((re.escape(k), v) for k, v in dictionary.items())
	pattern    = re.compile("|".join(dictionary.keys()))
	string     = pattern.sub(lambda m: dictionary[re.escape(m.group(0))], string)

	return string

def multireplace(string, replacements, ignore_case=False):
	"""
	Source: https://gist.github.com/bgusach/a967e0587d6e01e889fd1d776c5f3729

	Given a string and a replacement map, it returns the replaced string.

	:param str string: string to execute replacements on
	:param dict replacements: replacement dictionary {value to find: value to replace}
	:param bool ignore_case: whether the match should be case insensitive
	:rtype: str

	"""
	if not replacements:
		# Edge case that'd produce a funny regex and cause a KeyError
		return string

	# If case insensitive, we need to normalize the old string so that later a replacement
	# can be found. For instance with {"HEY": "lol"} we should match and find a replacement for "hey",
	# "HEY", "hEy", etc.
	if ignore_case:
		def normalize_old(s):
			return s.lower()

		re_mode = re.IGNORECASE

	else:
		def normalize_old(s):
			return s

		re_mode = 0

	replacements = {normalize_old(key): val for key, val in replacements.items()}

	# Place longer ones first to keep shorter substrings from matching where the longer ones should take place
	# For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
	# 'hey ABC' and not 'hey ABc'
	rep_sorted = sorted(replacements, key=len, reverse=True)
	rep_escaped = map(re.escape, rep_sorted)

	# Create a big OR regex that matches any of the substrings to replace
	pattern = re.compile("|".join(rep_escaped), re_mode)

	# For each match, look up the new string in the replacements, being the key the normalized old string
	return pattern.sub(lambda match: replacements[normalize_old(match.group(0))], string)
