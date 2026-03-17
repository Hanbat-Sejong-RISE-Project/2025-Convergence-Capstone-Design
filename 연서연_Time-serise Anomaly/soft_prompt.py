import math
import numpy as np
import pandas as pd
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from datasets import Dataset

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

random.seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용할 디바이스: {device}")

local_model_path = "./Llama-3.2-1B"
print(f"로컬 경로 '{local_model_path}'에서 모델과 토크나이저를 로딩합니다...")

tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 모델은 한 번만 로드 (float16 권장)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16
)

# 토크나이저 패딩 토큰 설정 (토큰화 전에 해도 되고 지금 해도 무방)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 데이터 로드 및 전처리 (원본과 동일)
train_df = pd.read_csv('dataset/mitbih_train.csv', header=None)
train_df.rename(columns={train_df.shape[1]-1: 'label'}, inplace=True)
train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 0 else 1)

X = train_df.drop(columns=['label'])
y = train_df['label']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def preprocess_for_llm(df: pd.DataFrame) -> list:
    multiplied_df = df * 1000
    integer_df = multiplied_df.astype(int)
    text_list = [' '.join(row.astype(str)) for index, row in integer_df.iterrows()]
    return text_list

X_train_texts = preprocess_for_llm(X_train)
X_val_texts = preprocess_for_llm(X_val)

train_data = {'text': X_train_texts, 'label': y_train.tolist()}
val_data = {'text': X_val_texts, 'label': y_val.tolist()}

train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)

def create_prompt(sample):
    prompt_template = "데이터: {ecg_data}\n분석 결과:"
    answer = "정상" if sample['label'] == 0 else "비정상"
    formatted_text = prompt_template.format(ecg_data=sample['text']) + f" {answer}"
    return {"text": formatted_text}

train_dataset = train_dataset.map(create_prompt)
val_dataset = val_dataset.map(create_prompt)

def tokenize_function(examples):
    out = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    out["labels"] = out["input_ids"].copy()
    return out

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

detailed_prompt_text = """당신은 심전도(ECG) 기록을 분석하는 전문 심장학자입니다. 주어진 ECG를 "정상(Normal)" 또는 "비정상(Abnormal)" 으로 분류하는 것이 당신의 임무입니다.

비정상(Abnormal) 기록은 다음 네 가지 유형 중 하나일 수 있습니다:

S (SVEB)

V (VEB)

F (Fusion)

Q (Unknown)

아래 예시와 정확히 동일한 형식으로 답변하세요.

정상 예시: Normal

비정상 예시: Abnormal - S (SVEB)

또 다른 비정상 예시: Abnormal - V (VEB)

이제 다음 ECG 기록을 분류하세요:
"""

# 기존 prompt "Classify this ECG recording as Normal or Abnormal. If Abnormal, specify the anomaly type."

# PEFT 설정 
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,
    prompt_tuning_init='TEXT',
    prompt_tuning_init_text=detailed_prompt_text,
    tokenizer_name_or_path="./Llama-3.2-1B",
)

peft_model = get_peft_model(model, peft_config)
peft_model.to(device)
peft_model.print_trainable_parameters()

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 정확한 스텝 계산
train_batch_size = 8
steps_per_epoch = math.ceil(len(tokenized_train_dataset) / train_batch_size)

training_args = TrainingArguments(
    output_dir="./soft_prompt_model/run1",
    num_train_epochs=3,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=8,
    warmup_steps=min(500, max(1, steps_per_epoch//2)), 
    weight_decay=0.01,
    logging_dir='./logs/run1',

    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",

    logging_steps=steps_per_epoch,
    eval_steps=steps_per_epoch,
    save_steps=steps_per_epoch,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./best_soft_prompt_model")