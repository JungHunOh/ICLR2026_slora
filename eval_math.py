import argparse
import json
import pdb
import jsonlines
import transformers
import torch
import util
import time
import sys
from datetime import timedelta

MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

invalid_outputs = []
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def process_results(doc, completion, answer):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if util.is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False
def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def truncate_at_stop_tokens(text, stop_tokens):
    earliest_pos = len(text)
    for token in stop_tokens:
        pos = text.find(token)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
    return text[:earliest_pos]

def test_hendrycks_math(model_path, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('promt =====', problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["instruction"])
            hendrycks_math_ins.append(temp_instr)
            solution = item['output']
            temp_ans = remove_boxed(util.last_boxed_only_string(solution))
            hendrycks_math_answers.append(temp_ans)

    print('total length ===', len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins[start:end]
    hendrycks_math_answers = hendrycks_math_answers[start:end]
    print('lenght ====', len(hendrycks_math_ins))
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )

    model.to("cuda")
    model.eval()

    total_batches = len(batch_hendrycks_math_ins)
    start_time = time.time()

    res_completions = []
    for batch_idx, batch_prompts in enumerate(batch_hendrycks_math_ins):
        # Tokenize and move inputs to device
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode and postprocess each output
        for output in outputs:
            gen_text = tokenizer.decode(output, skip_special_tokens=True)
            #gen_text = truncate_at_stop_tokens(gen_text, stop_tokens)
            res_completions.append(gen_text)

        # Added: Print progress and estimated finish time
        elapsed = time.time() - start_time
        batches_left = total_batches - (batch_idx + 1)
        est_remaining = (elapsed / (batch_idx + 1)) * batches_left
        est_finish = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + est_remaining))
        print(f"[Batch {batch_idx+1}/{total_batches}] Elapsed: {timedelta(seconds=int(elapsed))}, Estimated finish: {est_finish}")
    
    results = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        res = process_results(prompt, completion, prompt_answer)
        results.append(res)

    correct_count = sum(results)
    acc = correct_count / len(results)

    # Added: Final summary print
    print(f"\nFinal results:")
    print(f"Number of correct answers: {correct_count}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Number of invalid outputs: {len(invalid_outputs)}")

    # Added: Save results to file
    save_path = f"experiment/{model_path.split('/')[-2]}_math.txt"
    with open(save_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"Number of correct answers: {correct_count}\n")
        f_out.write(f"Accuracy: {acc:.4f}\n")
        f_out.write(f"Number of invalid outputs: {len(invalid_outputs)}\n")
    print(f"Results saved to {save_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=1)  # batch_size
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_hendrycks_math(model_path=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size)