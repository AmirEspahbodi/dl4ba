import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizerFast, BertForMaskedLM, pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
import os
import requests
import zipfile
import random


# --- GloVe Configuration ---
GLOVE_FILE_NAME = 'glove.6B.300d.txt'
GLOVE_EMBEDDING_DIM = 300
GLOVE_PATH = f'./{GLOVE_FILE_NAME}'
GLOVE_ZIP_URL = 'http://nlp.stanford.edu/data/glove.6B.zip'
GLOVE_LOCAL_ZIP_PATH = './glove.6B.zip'

# --- Model & Training Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32 
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
VALID_SET_SIZE = 0.2
MIN_CLASS_REPRESENTATION = 2


# --- GloVe File Handling ---
def ensure_glove_file_is_present():
    if os.path.exists(GLOVE_PATH):
        print(f"Found existing GloVe text file: {GLOVE_PATH}")
        return True
    if not os.path.exists(GLOVE_LOCAL_ZIP_PATH):
        print(f"GloVe zip file {GLOVE_LOCAL_ZIP_PATH} not found. Attempting to download from {GLOVE_ZIP_URL}...")
        try:
            response = requests.get(GLOVE_ZIP_URL, stream=True)
            response.raise_for_status()
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024
            with open(GLOVE_LOCAL_ZIP_PATH, 'wb') as file, tqdm(
                desc=f"Downloading {os.path.basename(GLOVE_ZIP_URL)}",
                total=total_size_in_bytes, unit='iB', unit_scale=True, unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    bar.update(len(data))
                    file.write(data)
            print(f"Successfully downloaded {GLOVE_LOCAL_ZIP_PATH}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading GloVe zip file: {e}")
            if os.path.exists(GLOVE_LOCAL_ZIP_PATH): os.remove(GLOVE_LOCAL_ZIP_PATH)
            return False
        except Exception as e:
            print(f"An unexpected error occurred during download: {e}")
            if os.path.exists(GLOVE_LOCAL_ZIP_PATH): os.remove(GLOVE_LOCAL_ZIP_PATH)
            return False
    else:
        print(f"Found existing GloVe zip file: {GLOVE_LOCAL_ZIP_PATH}")

    print(f"Attempting to extract {GLOVE_FILE_NAME} from {GLOVE_LOCAL_ZIP_PATH}...")
    try:
        with zipfile.ZipFile(GLOVE_LOCAL_ZIP_PATH, 'r') as zip_ref:
            if GLOVE_FILE_NAME in zip_ref.namelist():
                zip_ref.extract(GLOVE_FILE_NAME, path=os.path.dirname(GLOVE_PATH) or '.')
                print(f"Successfully extracted {GLOVE_FILE_NAME} to {GLOVE_PATH}")
                return True
            else:
                print(f"Error: {GLOVE_FILE_NAME} not found inside {GLOVE_LOCAL_ZIP_PATH}.")
                print(f"Available files: {zip_ref.namelist()}")
                return False
    except zipfile.BadZipFile:
        print(f"Error: {GLOVE_LOCAL_ZIP_PATH} is a bad zip file. Please delete it and try again.")
        return False
    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        return False

if not ensure_glove_file_is_present():
    print("Could not obtain GloVe file. Exiting.")
    exit()

def load_glove_vectors(glove_path, embedding_dim):
    print(f"Loading GloVe vectors from {glove_path} with dimension {embedding_dim}...")
    if not os.path.exists(glove_path):
        print(f"Error: GloVe file not found at {glove_path}.")
        return None
    word_to_vec = {}
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading GloVe"):
                values = line.split()
                word = values[0]
                try:
                    vector = np.asarray(values[1:], dtype='float32')
                    if len(vector) == embedding_dim:
                        word_to_vec[word] = vector
                except ValueError:
                    pass
    except Exception as e:
        print(f"An error occurred while reading the GloVe file: {e}")
        return None
    if not word_to_vec:
        print(f"No word vectors loaded from {glove_path}.")
        return None
    print(f"Successfully loaded {len(word_to_vec)} word vectors.")
    return word_to_vec

glove_vectors_map = load_glove_vectors(GLOVE_PATH, GLOVE_EMBEDDING_DIM)

unk_embedding = np.random.rand(GLOVE_EMBEDDING_DIM).astype('float32') # Random UNK if not in GloVe
if glove_vectors_map:
    if '[unk]' in glove_vectors_map: unk_embedding = glove_vectors_map['[unk]']
    elif 'unk' in glove_vectors_map: unk_embedding = glove_vectors_map['unk']
else:
    print("GloVe vectors map is empty or not loaded. Using random UNK embedding. Training will be affected.")
    # exit() # Critical error, might be best to exit



print("Loading BERT tokenizer...")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
print("Tokenizer loaded.")



print("Loading gcc_data.csv...")
dataset_path = '../datasets/gcc_data.csv'
if not os.path.exists(dataset_path):
    print(f"Warning: '{dataset_path}' not found. Trying 'gcc_data.csv' in current directory.")
    dataset_path = 'gcc_data.csv' # Fallback to current directory

try:
    dataset = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"Error: gcc_data.csv not found at '{dataset_path}' or in the current directory. Please place it correctly.")
    exit()

print(f"Original gcc_data shape: {dataset.shape}")

# Basic preprocessing for text columns
dataset['Summary'] = dataset['Summary'].fillna('').astype(str)
dataset['Description'] = dataset['Description'].fillna('').astype(str)
temp = []
for i in range(len(dataset['Summary'])):
    temp.append(
        f"Summary = {dataset['Summary'].iloc[i]} | Description = {dataset['Description'].iloc[i]}"
    )
dataset['text_input'] = temp

# Target variable: Assignee. Drop rows with missing Assignee.
dataset.dropna(subset=['Assignee'], inplace=True)
dataset['Assignee'] = dataset['Assignee'].astype(str) # Ensure assignee names are strings
print(f"Shape after dropping NA assignees: {dataset.shape}")

# Label Encoding for Assignee
assignee_encoder = LabelEncoder()
dataset['assignee_encoded'] = assignee_encoder.fit_transform(dataset['Assignee'])
NUM_ACTUAL_CLS = len(assignee_encoder.classes_)
print(f"Number of unique assignees (classes): {NUM_ACTUAL_CLS}")
if NUM_ACTUAL_CLS <= 1:
    print("Error: Only one or no classes found after encoding. Cannot train classifier.")
    exit()



# --- Advanced Data Augmentation: Contextual Word Replacement using Masked LM ---
print("\nInitializing components for Contextual Word Augmentation...")

mlm_model_name = 'bert-base-uncased'
unmasker = None # Initialize unmasker to None
try:
    # Using pipeline for easier Masked LM prediction
    # Attempt to use GPU if available, otherwise it should default to CPU.
    # The device parameter can explicitly be set to torch.device("cpu") if GPU issues persist.
    if torch.cuda.is_available():
        # Check if CUDA is truly functional beyond just being available
        try:
            torch.cuda.init() # Attempt to initialize CUDA
            print("CUDA seems available and initialized. Attempting to use GPU for pipeline.")
            unmasker = pipeline('fill-mask', model=mlm_model_name, tokenizer=mlm_model_name, top_k=5, device=0) # Use first GPU
        except Exception as cuda_init_error:
            print(f"CUDA available but initialization failed: {cuda_init_error}. Falling back to CPU for pipeline.")
            unmasker = pipeline('fill-mask', model=mlm_model_name, tokenizer=mlm_model_name, top_k=5, device=-1) # Explicitly CPU
    else:
        print("CUDA not available. Using CPU for pipeline.")
        unmasker = pipeline('fill-mask', model=mlm_model_name, tokenizer=mlm_model_name, top_k=5) # Defaults to CPU (device=-1)
    
    if unmasker:
        print(f"Masked LM pipeline ('{mlm_model_name}') initialized successfully on device: {unmasker.device}.")
except Exception as e:
    # This broad exception will catch OSErrors during pipeline init if CUDA libs are missing
    # or other initialization issues.
    print(f"Error initializing Masked LM pipeline: {e}")
    print("Data augmentation will be SKIPPED.")
    print("Common reasons: Missing CUDA libraries (if GPU attempted), Hugging Face model download issue (check internet), or other transformer/PyTorch setup problems.")
    unmasker = None # Ensure it's None if any error occurred

def contextual_word_replacement_augmentation(text, unmasker_pipeline, mask_token="[MASK]", num_augmentations=1, mask_ratio=0.15):
    if unmasker_pipeline is None:
        return [text]

    augmented_texts = []
    original_words = text.split()

    if len(original_words) < 5:
        return [text] * num_augmentations

    for _ in range(num_augmentations):
        words_to_augment = list(original_words)
        num_words_to_mask = max(1, int(len(words_to_augment) * mask_ratio))
        num_words_to_mask = min(num_words_to_mask, len(words_to_augment))
        
        # Get indices of words to be masked. We sort them to process from start to end
        # but the actual masking strategy inside the loop will handle context.
        indices_to_potentially_mask = sorted(random.sample(range(len(words_to_augment)), num_words_to_mask))

        temp_augmented_words = list(words_to_augment)

        for word_idx_to_mask in indices_to_potentially_mask:
            # Create a copy for masking this specific word, using the current state of temp_augmented_words
            current_context_words = list(temp_augmented_words)
            if word_idx_to_mask >= len(current_context_words): continue

            original_word_at_idx = current_context_words[word_idx_to_mask]
            current_context_words[word_idx_to_mask] = mask_token
            single_masked_input_text = " ".join(current_context_words)

            try:
                predictions = unmasker_pipeline(single_masked_input_text)
                chosen_replacement = None
                
                if predictions and isinstance(predictions, list):
                    # The pipeline with top_k returns a list of potential fills for each mask.
                    # If single_masked_input_text has one [MASK], predictions is like:
                    # [[{'score': ..., 'token_str': 'word1'}, {'score': ..., 'token_str': 'word2'}]]
                    # OR if multiple masks are handled differently by some model versions (less common for fill-mask):
                    # [{'score': ..., 'token_str': 'word1'}, ...] for the first mask
                    
                    # Assuming predictions[0] is a list of dicts for the first (and only) mask
                    potential_fills = predictions[0] if isinstance(predictions[0], list) else predictions

                    for pred_option in potential_fills:
                        if isinstance(pred_option, dict) and pred_option['token_str'].strip().lower() != original_word_at_idx.lower():
                            chosen_replacement = pred_option['token_str'].strip()
                            break
                    if not chosen_replacement and potential_fills: # Fallback to top prediction
                         if isinstance(potential_fills[0], dict):
                            chosen_replacement = potential_fills[0]['token_str'].strip()
                
                if chosen_replacement:
                    temp_augmented_words[word_idx_to_mask] = chosen_replacement # Update the main list for this augmentation
            except Exception as e:
                # print(f"Warning: Augmentation step failed for mask at index {word_idx_to_mask}. Error: {e}")
                pass 
        
        final_augmented_text = " ".join(temp_augmented_words)
        augmented_texts.append(final_augmented_text)

    return augmented_texts if augmented_texts else [text]


# --- Apply Augmentation ---
dataset_with_aug = dataset.copy()
dataset_with_aug['is_augmented'] = False

if unmasker:
    AUGMENTATION_TARGET_COUNT = int(dataset.shape[0] * 0.25) # Example: Augment to add 25% more samples
    NUM_AUGMENTATIONS_PER_SELECTED_SAMPLE = 1
    AUGMENTATION_MASK_RATIO = 0.10 # Mask 10% of words

    print(f"\nApplying contextual word replacement augmentation to generate approx. {AUGMENTATION_TARGET_COUNT} new samples...")
    new_rows = []
    
    num_original_samples_to_augment = min(AUGMENTATION_TARGET_COUNT, len(dataset))
    if num_original_samples_to_augment == 0 and AUGMENTATION_TARGET_COUNT > 0:
        print("Warning: No original samples available to augment from, but augmentation target > 0.")

    candidate_indices = dataset.index.tolist()
    if num_original_samples_to_augment < len(dataset):
         indices_to_augment = random.sample(candidate_indices, num_original_samples_to_augment)
    else:
         indices_to_augment = candidate_indices

    for original_idx in tqdm(indices_to_augment, desc="Augmenting samples"):
        row = dataset.loc[original_idx]
        original_text = row['text_input']
        assignee_label = row['assignee_encoded']
        original_summary, original_description, original_assignee, original_bug_id = row['Summary'], row['Description'], row['Assignee'], row['Bug_ID']

        augmented_texts = contextual_word_replacement_augmentation(
            original_text, unmasker, mask_token=unmasker.tokenizer.mask_token,
            num_augmentations=NUM_AUGMENTATIONS_PER_SELECTED_SAMPLE, mask_ratio=AUGMENTATION_MASK_RATIO
        )
        for i, aug_text in enumerate(augmented_texts):
            if aug_text != original_text: 
                new_rows.append({
                    'Bug_ID': f"{original_bug_id}_aug{i+1}", 'Assignee': original_assignee,
                    'Summary': original_summary, 
                    'Description': "AUGMENTED: " + (aug_text.split("| Description =")[-1].strip() if "| Description =" in aug_text else aug_text),
                    'Status': 'AUGMENTED_BUG_STATUS', 'text_input': aug_text,
                    'assignee_encoded': assignee_label, 'is_augmented': True
                })
    
    if new_rows:
        augmented_df = pd.DataFrame(new_rows)
        dataset_with_aug = pd.concat([dataset_with_aug, augmented_df], ignore_index=True)
        print(f"\nShape of dataset after augmentation: {dataset_with_aug.shape}")
        print(f"Number of augmented samples added: {len(augmented_df)}")
        
        if len(augmented_df) > 0 and indices_to_augment:
            print("\nExample of augmentation:")
            example_original_idx = indices_to_augment[0]
            print(f"Original text (from index {example_original_idx}): {dataset.loc[example_original_idx]['text_input']}")
            corresponding_aug_sample = augmented_df[augmented_df['Bug_ID'].str.startswith(str(dataset.loc[example_original_idx]['Bug_ID']) + "_aug")]
            if not corresponding_aug_sample.empty:
                 print(f"One augmented text for it: {corresponding_aug_sample.iloc[0]['text_input']}")
            else:
                 print(f"First augmented sample (could not find direct example match): {augmented_df.iloc[0]['text_input']}")
    else:
        print("\nNo augmented samples were effectively generated or added.")
else:
    print("\nSkipping augmentation as unmasker pipeline is not available (e.g., due to initialization error).")

# --- Post-Augmentation ---
print(f"\nFinal dataset size: {dataset_with_aug.shape}")

dataset_with_aug['assignee_encoded'] = dataset_with_aug['assignee_encoded'].astype(int)

if NUM_ACTUAL_CLS > 1:
    X = dataset_with_aug['text_input']
    y = dataset_with_aug['assignee_encoded']
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        print(f"Warning: Could not stratify during train_test_split due to: {e}. Splitting without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nShape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    if not y_train.empty:
        print(f"Train set class distribution (top 5 classes):\n{y_train.value_counts(normalize=True).nlargest(5)}")
    if not y_test.empty:
        print(f"Test set class distribution (top 5 classes):\n{y_test.value_counts(normalize=True).nlargest(5)}")
else:
    print("Skipping train_test_split as NUM_ACTUAL_CLS <= 1")

print("\nBaseline code with contextual augmentation integration is ready.")
print("Further steps would involve defining a model, training, and evaluation.")



# --- Train/Test Split ---
texts = dataset_with_aug['text_input'].tolist()
labels = dataset_with_aug['assignee_encoded'].tolist()

# Stratify only if all classes have at least MIN_CLASS_REPRESENTATION samples
assignee_counts = dataset_with_aug['Assignee'].value_counts()
small_classes = assignee_counts[assignee_counts < MIN_CLASS_REPRESENTATION].index.tolist()

stratify_labels = None
if not small_classes: 
    stratify_labels = labels
    print(f"Attempting stratified split. All {NUM_ACTUAL_CLS} classes have at least {MIN_CLASS_REPRESENTATION} samples.")
else:
    print(f"Warning: {len(small_classes)} classes have fewer than {MIN_CLASS_REPRESENTATION} samples. Proceeding without stratification.")
    print(f"Small classes: {small_classes[:5]}...")


train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=VALID_SET_SIZE, random_state=42, shuffle=True, stratify=stratify_labels
)

print(f"Train set size: {len(train_texts)}, Test set size: {len(test_texts)}")


# --- Prepare data for DataLoader (list of dicts) ---
def create_data_list(texts, labels):
    return [{'text': text, 'label': label} for text, label in zip(texts, labels)]

train_data_list = create_data_list(train_texts, train_labels)
test_data_list = create_data_list(test_texts, test_labels)


# --- Collate Function  ---
def collate_with_glove(batch, hf_tokenizer, glove_word_vectors, embedding_dimension, unk_word_embedding):
    labels_list = [item['label'] for item in batch]
    texts_list = [item['text'] for item in batch]

    # Ensure labels are LongTensors for CrossEntropyLoss
    labels = torch.LongTensor(labels_list)

    all_sequences_as_vecs = []
    for text_item in texts_list:
        string_tokens = hf_tokenizer.tokenize(str(text_item)) # Ensure text_item is string

        if not string_tokens:
            all_sequences_as_vecs.append(torch.tensor(unk_word_embedding, dtype=torch.float).unsqueeze(0))
            continue

        current_sequence_embeddings = []
        for token_str in string_tokens:
            vec = glove_word_vectors.get(token_str, unk_word_embedding)
            current_sequence_embeddings.append(torch.tensor(vec, dtype=torch.float))

        if not current_sequence_embeddings:
            all_sequences_as_vecs.append(torch.tensor(unk_word_embedding, dtype=torch.float).unsqueeze(0))
        else:
            all_sequences_as_vecs.append(torch.stack(current_sequence_embeddings))

    vecs_padded = pad_sequence(all_sequences_as_vecs, batch_first=False, padding_value=0.0)
    return vecs_padded, labels


# --- Create DataLoaders ---
if glove_vectors_map is not None and train_data_list and test_data_list:
    collate_fn_custom = partial(collate_with_glove,
                                 hf_tokenizer=tokenizer,
                                 glove_word_vectors=glove_vectors_map,
                                 embedding_dimension=GLOVE_EMBEDDING_DIM,
                                 unk_word_embedding=unk_embedding)

    train_loader = DataLoader(train_data_list, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_custom)
    test_loader = DataLoader(test_data_list, batch_size=BATCH_SIZE, collate_fn=collate_fn_custom)
    print("DataLoaders created for gcc_data.")
else:
    print("GloVe vectors not loaded or data lists are empty. Cannot create DataLoaders.")
    train_loader = None
    test_loader = None


# --- Utility Classes and CNN Model (largely unchanged) ---
class AverageMeter(object):
    def __init__(self): self.reset()
    def reset(self): self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

class CNNModel1(nn.Module):
    def __init__(self, embed_dim, filter_sizes, num_filters_per_size, num_classes, dropout_rate):
        super(CNNModel1, self).__init__()
        self.conv1d_list = nn.ModuleList(
            [
                nn.Conv1d(in_channels=embed_dim,
                          out_channels=n_filters,
                          kernel_size=f_size)
                for f_size, n_filters in zip(filter_sizes, num_filters_per_size)
            ]
        )
        # Calculate total number of filters correctly
        total_filters = sum(num_filters_per_size)
        self.fc = nn.Linear(total_filters, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # x shape: (max_seq_length, batch_size, embedding_dim)
        x = x.permute(1, 2, 0) # (batch_size, embedding_dim, max_seq_length)

        conv_outputs = []
        for conv_layer in self.conv1d_list:
            conv_output = conv_layer(x)
            conv_output = F.relu(conv_output)
            conv_output = F.max_pool1d(conv_output, kernel_size=conv_output.size(2)).squeeze(2)
            conv_outputs.append(conv_output)

        x_concatenated = torch.cat(conv_outputs, dim=1)
        x_dropped_out = self.dropout(x_concatenated)
        logits = self.fc(x_dropped_out)
        return logits

class CNNModel2(nn.Module):
    def __init__(self, 
                    embed_dim,
                    filter_sizes,
                    num_filters_per_size,
                    num_classes,
                    dropout_rate,
                    hidden_dim_fc
                 ):
        super(CNNModel2, self).__init__()
        self.conv1d_list = nn.ModuleList(
            [
                nn.Conv1d(in_channels=embed_dim,
                          out_channels=n_filters,
                          kernel_size=f_size)
                for f_size, n_filters in zip(filter_sizes, num_filters_per_size)
            ]
        )
        # Calculate total number of filters correctly
        total_filters = sum(num_filters_per_size)
        self.fc1 = nn.Linear(total_filters, hidden_dim_fc)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim_fc, num_classes)

    def forward(self, x):
        # x shape: (max_seq_length, batch_size, embedding_dim)
        x = x.permute(1, 2, 0) # (batch_size, embedding_dim, max_seq_length)

        conv_outputs = []
        for conv_layer in self.conv1d_list:
            conv_output = conv_layer(x)
            conv_output = F.relu(conv_output)
            conv_output = F.max_pool1d(conv_output, kernel_size=conv_output.size(2)).squeeze(2)
            conv_outputs.append(conv_output)

        x_concatenated = torch.cat(conv_outputs, dim=1)
        x_dropped_out1 = self.dropout(x_concatenated)
        x_fc1 = F.relu(self.fc1(x_dropped_out1))
        x_dropped_out2 = self.dropout(x_fc1)
        logits = self.fc2(x_dropped_out2)
        return logits


# --- Training and Validation Functions (largely unchanged) ---
def train_one_epoch(model, dataloader, loss_function, optim, current_epoch=None):
    model.train()
    loss_meter = AverageMeter()
    correct_predictions = 0
    total_samples = 0
    tepoch = tqdm(dataloader, unit="batch")
    if current_epoch is not None: tepoch.set_description(f"Epoch {current_epoch+1}")
    for inputs, targets in tepoch:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        optim.zero_grad(); loss.backward(); optim.step()
        loss_meter.update(loss.item(), inputs.size(1)) # batch_size is second dim of permuted input
        _, predicted_labels = torch.max(outputs.data, 1)
        total_samples += targets.size(0)
        correct_predictions += (predicted_labels == targets).sum().item()
        tepoch.set_postfix(loss=loss_meter.avg, accuracy=100. * correct_predictions / total_samples if total_samples > 0 else 0)
    return model, loss_meter.avg, (100. * correct_predictions / total_samples if total_samples > 0 else 0)

def validate_model(model, dataloader, loss_function):
    model.eval()
    loss_meter = AverageMeter()
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad(), tqdm(dataloader, unit="batch", desc="Validating") as tepoch:
        for inputs, targets in tepoch:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss_meter.update(loss.item(), inputs.size(1))
            _, predicted_labels = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted_labels == targets).sum().item()
            tepoch.set_postfix(loss=loss_meter.avg, accuracy=100. * correct_predictions / total_samples if total_samples > 0 else 0)
    return loss_meter.avg, (100. * correct_predictions / total_samples if total_samples > 0 else 0)


EMBEDDING_DIM_VALUE = 300
N_FILTERS_LIST = [512, 512, 512]
FILTER_SIZES_LIST = [3, 4, 5]
OUTPUT_DIM_VALUE = NUM_ACTUAL_CLS
DROPOUT_RATE_VALUE = 0.5
HIDDEN_DIM_FC_VALUE = 256

print(f"\nUsing device: {DEVICE}")
# Test one batch
try:
    x_batch, y_batch = next(iter(train_loader))
    print(f"\nSample batch - X shape: {x_batch.shape}, Y shape: {y_batch.shape}")
except Exception as e:
    print(f"Error getting a batch from train_loader (might be empty or an issue with collate_fn): {e}")
    # exit() # Exit if we can't even get a batch

# Initialize model, loss, optimizer
cnn_model = CNNModel2(
                        embed_dim=GLOVE_EMBEDDING_DIM,
                        filter_sizes=FILTER_SIZES_LIST,
                        num_filters_per_size=N_FILTERS_LIST,
                        num_classes=NUM_ACTUAL_CLS, # Use dynamic number of classes
                        dropout_rate=DROPOUT_RATE_VALUE,
                        hidden_dim_fc=HIDDEN_DIM_FC_VALUE
                    ).to(DEVICE)

print(f"\nCNN Model Initialized with {NUM_ACTUAL_CLS} output classes.")
print(cnn_model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# optimizer = optim.SGD(cnn_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9)


history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': []}
best_valid_loss = float('inf')

print("\n--- Starting Model Training ---")
for epoch_idx in range(NUM_EPOCHS):
    cnn_model, train_loss, train_acc = train_one_epoch(cnn_model, train_loader, criterion, optimizer, epoch_idx)
    valid_loss, valid_acc = validate_model(cnn_model, test_loader, criterion)
    history['train_loss'].append(train_loss); history['valid_loss'].append(valid_loss)
    history['train_acc'].append(train_acc); history['valid_acc'].append(valid_acc)
    if valid_loss < best_valid_loss:
        torch.save(cnn_model.state_dict(), 'cnn_bug_assign_best.pt')
        best_valid_loss = valid_loss
        print('Model Saved as cnn_bug_assign_best.pt!')
    print(f'Validation: Loss = {valid_loss:.4f}, Accuracy = {valid_acc:.2f}%')
    print("-" * 30)
print("--- Training Finished ---")

if history['train_loss']:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history['train_loss'], label='Train Loss'); plt.plot(history['valid_loss'], label='Valid Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss vs. Epochs')
    plt.subplot(1, 2, 2); plt.plot(history['train_acc'], label='Train Accuracy'); plt.plot(history['valid_acc'], label='Valid Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.title('Accuracy vs. Epochs')
    plt.tight_layout(); plt.show()

print("\nScript execution completed.")
