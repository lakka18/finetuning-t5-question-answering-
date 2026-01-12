# finetuning-t5-question-answering-

---
Name : Arthur Trageser
NIM  : 1103223090

# Task 2: Question Answering with T5-base on SQuAD

## Overview

This project implements an end-to-end Question Answering (QA) system using T5-base, an encoder-decoder Transformer architecture. The model is fine-tuned on the Stanford Question Answering Dataset (SQuAD v1.1) to generate answer text directly from context paragraphs and questions.

**Key Details:**
- **Model:** T5-base (220M parameters)
- **Dataset:** SQuAD v1.1 (87,599 training samples)
- **Task:** Generative Question Answering
- **Evaluation Metrics:** Exact Match (EM) and F1 Score

---

## Dataset: SQuAD v1.1

### Statistics
- **Training samples:** 10000
- **Validation samples:** 1000
- **Source:** `rajpurkar/squad` from HuggingFace Datasets

### Data Structure
Each example contains:
- `context`: A paragraph containing the answer
- `question`: A question about the context
- `answers`: Dictionary with answer text and start position

### Data Analysis Insights

The notebook includes comprehensive exploratory data analysis:

1. **Context Length Distribution:**
   - Most contexts range from 50-150 words
   - Median: ~120 words
   - Some contexts exceed 400 words

2. **Question Length Distribution:**
   - Typical questions: 5-15 words
   - Questions are concise and focused

3. **Answer Length Distribution:**
   - Most answers: 1-5 words
   - Answers are typically short phrases or named entities

4. **Answer Position Analysis:**
   - Answers distributed throughout contexts
   - No strong bias toward beginning or end
   - Slight concentration in middle portions

5. **Correlation Analysis:**
   - Weak correlation between context length and answer length
   - Question length independent of context complexity

---

## Model Architecture

### T5-base Specifications
- **Type:** Encoder-Decoder Transformer
- **Parameters:** 220 million
- **Architecture:** 12 encoder layers + 12 decoder layers
- **Hidden size:** 768
- **Attention heads:** 12 per layer

### Text-to-Text Format
T5 treats all NLP tasks as text-to-text:
```
Input:  "question: {question} context: {context}"
Output: "{answer_text}"
```

---

## Implementation

### Hyperparameters

```python
MODEL_NAME = "t5-base"
MAX_INPUT_LENGTH = 512      # tokens for question + context
MAX_TARGET_LENGTH = 32      # tokens for answer
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
```

### Training Configuration
- **Optimizer:** AdamW
- **Learning rate schedule:** Linear warmup + decay
- **Evaluation:** Every 2000 steps
- **Generation:** Beam search with num_beams=4
- **Hardware:** GPU (L4 or T4 recommended)
- **Training time:** ~2-3 hours per epoch

### Preprocessing Pipeline

1. Combine question and context with task prefix
2. Tokenize inputs (truncate to 512 tokens)
3. Tokenize target answers (truncate to 32 tokens)
4. Replace padding tokens with -100 in labels (ignored in loss)

---

## Results

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | Percentage of predictions matching ground truth exactly |
| **F1 Score** | Token-level overlap between prediction and reference |

The model achieves competitive performance on SQuAD validation set. F1 scores are typically 5-10 points higher than Exact Match due to partial credit for overlapping tokens.

### Training Dynamics

**Loss Curves:**
- Steady decrease in training loss throughout training
- Validation loss follows similar trend
- Minimal overfitting in single epoch

**Metric Evolution:**
- Both EM and F1 improve consistently during training
- Metrics show steady improvement until plateau near end of epoch

---

## Usage

### Setup (Google Colab)

The notebook is designed for Google Colab with GPU acceleration:

1. Open the notebook in Colab
2. Enable GPU: `Runtime → Change runtime type → GPU`
3. Run the setup cell (installs all dependencies)
4. Mount Google Drive (optional, for saving models)

### Setup (Local)

```bash
pip install transformers datasets torch accelerate tensorboard evaluate
```

### Running the Notebook

Execute cells sequentially:
1. **Setup and Installation** - Install dependencies
2. **Load and Explore Dataset** - Download SQuAD and analyze
3. **Tokenization and Preprocessing** - Prepare data for training
4. **Model Training** - Fine-tune T5-base (~2-3 hours)
5. **Model Evaluation** - Compute metrics and visualizations

### Using the Trained Model

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class T5QA:
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint_path).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)
        self.model.eval()
    
    def answer(self, question, context):
        input_text = f"question: {question} context: {context}"
        input_ids = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids, 
                max_length=32, 
                num_beams=4, 
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load model
qa = T5QA("/path/to/checkpoints")

# Make predictions
question = "What is the capital of France?"
context = "Paris is the capital of France with 2.2 million people."
answer = qa.answer(question, context)
print(f"Answer: {answer}")
```

---

## Project Structure

```
task2_squad/
├── notebooks/
│   └── task2_squad.ipynb          # Main training notebook
├── checkpoints/                    # Saved model weights
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── spiece.model
├── reports/
│   └── training_results.csv        # Final evaluation metrics
├── output/                         # Temporary training outputs
└── logs/                          # TensorBoard logs
```

---

## Sample Predictions

### Example 1
```
Question: What is the capital of France?
Context:  Paris is the capital of France with 2.2 million people.
Answer:   Paris
Status:   ✅ Exact Match
```

### Example 2
```
Question: When was the Eiffel Tower built?
Context:  The Eiffel Tower was built between 1887 and 1889 for the World's Fair.
Answer:   1887 and 1889
Status:   ⚠️ Partial Match (high F1, no exact match)
```

---

## Visualizations

The notebook generates several visualizations:

1. **Dataset Statistics:**
   - Context length histogram
   - Question length histogram
   - Answer length histogram
   - Answer position distribution

2. **Training Progress:**
   - Training vs validation loss curves
   - Exact Match progression
   - F1 score progression

3. **Correlation Analysis:**
   - Correlation matrix between text lengths

---

## Requirements

### Python Packages
```
transformers>=4.30.0
datasets>=2.12.0
torch>=2.0.0
accelerate>=0.20.0
tensorboard>=2.13.0
evaluate>=0.4.0
numpy
pandas
matplotlib
seaborn
```

### Hardware
- **GPU:** Required for efficient training (8GB+ VRAM recommended)
- **RAM:** 16GB+ system memory
- **Storage:** ~2GB for model checkpoints

---

## Future Improvements

1. **Extended Training:** Train for 3-5 epochs for better convergence
2. **Longer Contexts:** Experiment with 768 or 1024 token inputs
3. **Model Variants:** Try T5-large or Flan-T5 for improved performance
4. **Data Augmentation:** Add paraphrased questions for robustness
5. **Multi-answer Support:** Handle questions with multiple valid answers
6. **Error Analysis:** Analyze failure cases systematically
---

## License

This project is for educational purposes as part of a Deep Learning for NLP course assignment.

---
