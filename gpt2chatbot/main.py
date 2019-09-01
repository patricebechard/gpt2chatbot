import logging
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

logging.basicConfig(level=logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TOKENS_TO_GENERATE = 100
K_VALUE = 10
P_VALUE = 0.01


def greedy_decode(prediction_scores):
    """
    We only select the max probability of the prediction scores
    """
    return torch.argmax(prediction_scores).cpu().item()


def top_k_decode(prediction_scores, k=10):
    """
    We sample uniformly from the k words with top probability.
    """
    values, indexes = torch.topk(prediction_scores, k=k)
    choice = indexes[torch.randint(0, indexes.size(0), (1,))].cpu().item()
    return choice


def nucleus_decode(prediction_scores, p=0.02):
    """
    We sample uniformly from words with probability > p.
    """
    prediction_probs = F.softmax(prediction_scores, dim=0)
    indexes = torch.nonzero(prediction_probs >= p).flatten()
    choice = indexes[torch.randint(0, indexes.size(0), (1,))].cpu().item()
    return choice

def categorical_decode(prediction_scores):
    """
    We sample directly from the prediction scores. Tail of the distribution is 
    thus not excluded from sampling, but chance of sampling is of lower probability
    """
    probs = Categorical(logits=prediction_scores)
    choice = probs.sample().cpu().item()
    return choice


if __name__ == "__main__":

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)

    # prompt = input("Enter a prompt : ")
    prompt = "the blue cat is yellow"
    prompt = tokenizer.encode(prompt)

    with torch.no_grad():
        for _ in tqdm(range(NUM_TOKENS_TO_GENERATE)):

            input_ids = torch.LongTensor(prompt).unsqueeze(0).to(DEVICE)

            outputs = model(input_ids)
            prediction_scores, past = outputs[:2]
            next_word_prediction_scores = prediction_scores[0, -1]

            # greedy decoding
            # prediction = greedy_decode(next_word_prediction_scores)
            # prediction = top_k_decode(next_word_prediction_scores, k=K_VALUE)
            # prediction = nucleus_decode(next_word_prediction_scores, p=P_VALUE)
            prediction = categorical_decode(next_word_prediction_scores)
            prompt.append(prediction)

    # convert prompt to full string
    output = tokenizer.decode(prompt)

    print("Output : ")
    print(output)


