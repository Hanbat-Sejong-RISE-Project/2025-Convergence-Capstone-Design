import argparse
import pandas as pd
import numpy as np
import os
import random
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import re
import sys

random.seed(42)
np.random.seed(42)

def choose_dtype():
    if hasattr(torch, "bfloat16") and torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0], dtype=torch.bfloat16, device="cuda")
            return torch.bfloat16
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32

def preprocess_for_llm(df: pd.DataFrame) -> list:
    multiplied_df = df * 1000
    integer_df = multiplied_df.astype(int)
    text_list = [' '.join(row.astype(str)) for index, row in integer_df.iterrows()]
    return text_list

def parse_response(response: str) -> int:
    if not response or not response.strip():
        return 0

    text = response.lower()
    if "분석 결과:" in text:
        answer_part = text.split("분석 결과:")[-1].strip()
    else:
        answer_part = text.strip()

    answer_part = re.sub(r'\s+', ' ', answer_part)

    anomaly_keywords = ["비정상", "이상", "부정맥", "anomaly", "abnormal", "outlier", "irregular"]
    normal_keywords = ["정상", "정상적", "normal", "no anomaly", "정상입니다", "이상없음", "regular"]

    if any(kw in answer_part for kw in anomaly_keywords):
        return 1
    if any(kw in answer_part for kw in normal_keywords):
        return 0

    return 0

def plot_and_save_curves(y_true, y_scores, output_dir="./evaluation_results"):
    os.makedirs(output_dir, exist_ok=True)

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    pr_path = os.path.join(output_dir, "pr_curve.png")
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    print(f"[저장 완료] PR Curve: {pr_path}")
    plt.close()

    print(f"Average Precision (AP): {avg_precision:.4f}")
    
    return avg_precision

def main(args):
    model_path = args.model_path
    peft_dir = args.peft_dir
    test_csv = args.test_csv
    max_new_tokens = args.max_new_tokens

    if torch.cuda.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        print("경고: CUDA 사용 불가 -> CPU로 실행합니다.")
        device = torch.device("cpu")

    dtype = choose_dtype()
    print(f"선택된 dtype: {dtype}")

    print("토크나이저 로드중:", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    print("베이스 모델 로드중:", model_path)
    model_kwargs = dict(device_map="auto", torch_dtype=dtype)
    try:
        base_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except Exception as e:
        print("모델 로드 실패:", e)
        print("device_map='auto' 대신 device_map=None과 map_location을 시도합니다.")
        base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=None).to(torch.device("cpu"))

    if peft_dir:
        print("PEFT 로드중:", peft_dir)
        try:
            model = PeftModel.from_pretrained(base_model, peft_dir, device_map="auto")
        except Exception as e:
            print("PEFT 적용 중 오류:", e)
            print("PEFT 없이 base model 사용합니다.")
            model = base_model
    else:
        model = base_model

    model.eval()


    print("[INFO] 테스트 CSV 로드중:", test_csv)
    df = pd.read_csv(test_csv, header=None)
    label_col = df.shape[1] - 1
    df.rename(columns={label_col: 'label'}, inplace=True)
    df['label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)

    X_test_texts = preprocess_for_llm(df.drop(columns=['label']))
    y_test_true = df['label'].tolist()
    print("[INFO] 테스트 데이터 준비 완료! 샘플 수:", len(X_test_texts))

    prompt_template_inference = "데이터: {ecg_data}\n분석 결과:"

    raw_predictions = []
    prediction_probs = []
    device = next(model.parameters()).device
    print("추론 시작 (device):", device)

    pbar = tqdm(X_test_texts,
                desc="추론 진행 중",
                file=sys.stdout,    
                leave=False,
                dynamic_ncols=False,   
                mininterval=0.5)

    for data_point in pbar:
        prompt = prompt_template_inference.format(ecg_data=data_point)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,  
                return_dict_in_generate=True  
            )
            
            # 생성된 텍스트
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs.sequences[0, input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            full_response_for_parsing = f"분석 결과: {generated_text}"
            raw_predictions.append(full_response_for_parsing)
            

            if len(outputs.scores) > 0:
                first_token_logits = outputs.scores[0][0]  # 첫 번째 생성 토큰의 logits
                probs = torch.softmax(first_token_logits, dim=-1)
                
                normal_token_id = tokenizer.encode("정상", add_special_tokens=False)[0]
                abnormal_token_id = tokenizer.encode("비정상", add_special_tokens=False)[0]
                
                normal_prob = probs[normal_token_id].item()
                abnormal_prob = probs[abnormal_token_id].item()
                
                total_prob = normal_prob + abnormal_prob
                if total_prob > 0:
                    abnormal_score = abnormal_prob / total_prob
                else:
                    abnormal_score = float(parse_response(full_response_for_parsing))
                
                prediction_probs.append(abnormal_score)
            else:
                parsed_label = parse_response(full_response_for_parsing)
                prediction_probs.append(float(parsed_label))

    pbar.close()

    print("추론 완료!")
    threshold = 0.5

    y_scores = np.array(prediction_probs, dtype=float)  
    y_pred = (y_scores >= threshold).astype(int)        

    print(f"\n성능 리포트:")
    report = classification_report(
        y_test_true,
        y_pred,
        target_names=["정상 (class 0)", "비정상 (class 1)"],
    )
    print(report)

    
    # PR/ROC curve 그리기 및 저장
    output_dir = os.path.join(os.path.dirname(peft_dir) if peft_dir else "./", "evaluation_results_2")
    plot_and_save_curves(y_test_true, y_scores, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 기반 ECG 이상탐지 - inference 스크립트")
    parser.add_argument("--model-path", type=str, required=True,
                        help="로컬 또는 허브의 base 모델 경로 (예: ./Llama-3.2-1B)")
    parser.add_argument("--peft-dir", type=str, default=None,
                        help="(선택) PEFT로 학습한 디렉터리 경로 (예: ./best_soft_prompt_model). 없으면 base만 사용.")
    parser.add_argument("--test-csv", type=str, default="dataset/mitbih_test.csv",
                        help="테스트 CSV 파일 경로")
    parser.add_argument("--max-new-tokens", type=int, default=12,
                        help="생성 토큰 수")
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print("\n사용자 종료 (Ctrl-C)")
        sys.exit(0)


# CUDA_VISIBLE_DEVICES=2 python run_inference.py --model-path ./Llama-3.2-1B --peft-dir ./best_soft_prompt_model 
# CUDA_VISIBLE_DEVICES=3 python run_inference.py --model-path ./Llama-3.2-1B --peft-dir ./best_lora_model 