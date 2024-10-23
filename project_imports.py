# Standard Library Imports
import ast
import copy
import csv
import json
import math
import os
import re
import time
import warnings
import logging
import random
import collections
import gc
from collections import Counter, defaultdict
from typing import List, Tuple, Optional
from IPython.display import HTML, display
import math
import time
from unidecode import unidecode
import string
import multiprocessing as mp



# Data Handling Libraries
import numpy as np
import pandas as pd
import csv
from torch.utils.data import random_split
from torch.cuda.amp import GradScaler 
import datasets
from datasets import ClassLabel, Sequence, Dataset, DatasetDict, load_dataset, load_metric, concatenate_datasets, load_from_disk


# Data Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
# import scikitplot as skplt  # Uncomment if scikit-plot is installed and needed

# Machine Learning: Model Preparation
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, f1_score
from sklearn.model_selection import cross_val_score, cross_validate, KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

# Machine Learning: Models and Frameworks
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
import evaluate
import xgboost
#import wandb
from xgboost import plot_importance  # Uncomment if xgboost importance plot is required


# NLP and Transformers
import spacy
import transformers
from transformers import (AdamW, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForMultipleChoice,
                          AutoTokenizer, CamembertForSequenceClassification, DistilBertConfig,
                          DistilBertForSequenceClassification, DistilBertModel, EarlyStoppingCallback,
                          get_linear_schedule_with_warmup, RobertaForSequenceClassification, EvalPrediction,
                          Trainer, TrainerCallback, TrainingArguments, XLMRobertaForSequenceClassification,
                         DefaultDataCollator, BertForQuestionAnswering, DataCollatorWithPadding, PreTrainedTokenizerFast,
                         default_data_collator, is_torch_xla_available, pipeline)
from transformers.trainer_utils import PredictionOutput, speed_metrics

# Experiment Tracking and Optimization Utilities
import optuna
from optuna.trial import TrialState
# import wandb  # Uncomment if using Weights & Biases for experiment tracking

# Progress Bar Utilities
from tqdm.auto import tqdm


print("Project libraries imported!")