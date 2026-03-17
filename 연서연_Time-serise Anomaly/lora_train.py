import numpy as np
import os, re
import pandas as pd
import random
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 기본

random.seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용할 디바이스: {device}")

# === 사용자 설정 ===
local_model_path = "./Llama-3.2-1B"
train_csv_path = "dataset/mitbih_train.csv"

print(f"로컬 경로 '{local_model_path}'에서 모델과 토크나이저를 로딩합니다...")


tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=True)
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    local_model_path,             
    torch_dtype=torch_dtype
).to(device)

model.config.use_cache = False

print("로컬 모델 로딩 성공!")

train_df = pd.read_csv(train_csv_path, header=None)
train_df.rename(columns={train_df.shape[1]-1: 'label'}, inplace=True)
train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 0 else 1)

X = train_df.drop(columns=['label'])
y = train_df['label']

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

def preprocess_for_llm(df: pd.DataFrame) -> list:
    multiplied_df = df * 1000
    integer_df = multiplied_df.astype(int)
    text_list = [' '.join(row.astype(str)) for index, row in integer_df.iterrows()]
    return text_list

X_train_texts = preprocess_for_llm(X_train)
X_val_texts = preprocess_for_llm(X_val)

y_train = y_train.tolist()
y_val = y_val.tolist()

print("전처리 완료")

train_data = {'text': X_train_texts, 'label': y_train}
val_data = {'text': X_val_texts, 'label': y_val}

train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)

def create_prompt(example):
    prompt_template = "데이터: {ecg_data}\n분석 결과:"
    answer = "정상" if example['label'] == 0 else "비정상"
    formatted_text = prompt_template.format(ecg_data=example['text']) + f" {answer}"
    return {"text": formatted_text}

train_dataset = train_dataset.map(create_prompt)
val_dataset = val_dataset.map(create_prompt)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    out = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    out["labels"] = out["input_ids"].copy()
    return out

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

num_layers = len(model.model.layers)

lora_cfg_step = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules= ["gate_proj"],  
    # ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    bias="none"
)

profile = "step42`"
run_name = f"{profile}_r{lora_cfg_step.r}_seed42"

base_dir = "./save_lora_r4"
save_root = os.path.join(base_dir, run_name)
os.makedirs(save_root, exist_ok=True)

peft_model_lora = get_peft_model(model, lora_cfg_step)

# N ~ M-1번 레이어 학습
layers_to_train = set(range(8, 16))  

for name, param in peft_model_lora.named_parameters():
    if "lora" in name.lower():
        match = re.search(r"layers\.(\d+)\.", name)
        if match:
            layer_idx = int(match.group(1))
            param.requires_grad = layer_idx in layers_to_train

print("--- LoRA 모델의 훈련 가능 파라미터 ---")
peft_model_lora.print_trainable_parameters()

training_args_lora = TrainingArguments(
    output_dir= save_root,
    num_train_epochs= 3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,   
    warmup_steps=300,
    weight_decay=0.01,
    logging_dir= save_root,
    logging_strategy="epoch",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),  
    dataloader_pin_memory=True,
    seed=42
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer_lora = Trainer(
    model=peft_model_lora,
    args=training_args_lora,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
)

print("\n LoRA 튜닝을 시작합니다.")
trainer_lora.train()

trainer_lora.save_model(save_root)
tokenizer.save_pretrained(save_root)
print("Best checkpoint:", trainer_lora.state.best_model_checkpoint)
print(f"[완료] 전체 모델/토크나이저 저장: {save_root}")
