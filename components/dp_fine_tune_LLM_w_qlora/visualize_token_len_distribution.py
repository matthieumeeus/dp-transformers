import transformers
import datasets


dataset = datasets.DatasetDict({
            "train": datasets.Dataset.from_json(str("./components/dp_fine_tune_LLM/train.json")),
        })

tokenizer = transformers.AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

dataset = dataset.map(
            lambda x: {"chat": 
                       [tokenizer.apply_chat_template([{"role": "system", "content": ""}, 
                                                       {"role": "user", "content": q}, 
                                                       {"role": "assistant", "content": s}], tokenize=True) 
                                                       for q, s in zip(x["prompt"], x["completion"])]},
            batched=True,
            num_proc=None,
        )

input_lengths = [len(x) for x in dataset["train"]["chat"]]
input_lengths.sort(reverse=True)

# Plot sequence Lengths.
import seaborn as sns
sns.pairplot(input_lengths, kde=True).set(title='Distribution of Text Sequence Lengths')
