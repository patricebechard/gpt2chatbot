import os
import torch
from tqdm import tqdm
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
from gpt2chatbot.main import greedy_decode, top_k_decode, nucleus_decode, categorical_decode

K_VALUE = 25
P_VALUE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_COMPLETION_PER_TYPE = 3
NUM_TOKENS_TO_GENERATE = 800

if __name__ == "__main__":

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium", do_lower_case=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(DEVICE)

    prompt_types = ["trump", "basketball", "pizza", "kanye"]
    decoding_methods = ["greedy", "top_k", "top_k_weighted", "nucleus", "nucleus_weighted", "categorical"]

    prompts_path = "prompts"
    out_path = "out"

    for prompt_type in prompt_types:
        with open(os.path.join(prompts_path, prompt_type + ".txt")) as f:
            prompt = f.read()
        
        for decoding_method in decoding_methods:
            for completion_ix in range(NUM_COMPLETION_PER_TYPE):
                print(f"Generating sample for {prompt_type}, {decoding_method}, iter {completion_ix}")
                tokenized_prompt = tokenizer.encode(prompt)
                full = [elem for elem in tokenized_prompt]

                with torch.no_grad():
                    for _ in tqdm(range(NUM_TOKENS_TO_GENERATE)):
                        input_ids = torch.LongTensor(full).unsqueeze(0).to(DEVICE)
                        outputs = model(input_ids)
                        prediction_scores, past = outputs[:2]
                        next_word_prediction_scores = prediction_scores[0, -1]

                        if decoding_method == "greedy":
                            prediction = greedy_decode(next_word_prediction_scores)
                        elif decoding_method == "top_k":
                            prediction = top_k_decode(next_word_prediction_scores, k=K_VALUE)
                        elif decoding_method == "top_k_weighted":
                            prediction = top_k_decode(next_word_prediction_scores, k=K_VALUE, weighted=True)
                        elif decoding_method == "nucleus":
                            prediction = nucleus_decode(next_word_prediction_scores, p=P_VALUE)
                        elif decoding_method == "nucleus_weighted":
                            prediction = nucleus_decode(next_word_prediction_scores, p=P_VALUE)     
                        elif decoding_method == "categorical":
                            prediction = categorical_decode(next_word_prediction_scores)                       

                        full.append(prediction)

                # convert prompt to full string
                completion = tokenizer.decode(full[len(tokenized_prompt):])

                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                out_fname = os.path.join(out_path, f"{prompt_type}_{decoding_method}_{completion_ix}.txt")
                with open(out_fname, "w") as f:
                    f.write("PROMPT :\n")
                    f.write(prompt + "\n\n")
                    f.write("COMPLETION : \n")
                    f.write(completion + "\n")

