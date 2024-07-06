from transformers import T5Tokenizer, T5ForConditionalGeneration, DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

# Load the pre-trained T5 model and tokenizer for question generation
qg_model_name = 'valhalla/t5-small-qg-hl'
qg_tokenizer = T5Tokenizer.from_pretrained(qg_model_name)
qg_model = T5ForConditionalGeneration.from_pretrained(qg_model_name)

# Load the pre-trained DistilBERT model and tokenizer for question answering
qa_model_name = 'distilbert-base-uncased-distilled-squad'
qa_tokenizer = DistilBertTokenizer.from_pretrained(qa_model_name)
qa_model = DistilBertForQuestionAnswering.from_pretrained(qa_model_name)

def generate_qas(paragraph):
    # Encode the input paragraph for question generation
    input_text = f"generate questions: {paragraph}"
    input_ids = qg_tokenizer.encode(input_text, return_tensors="pt")

    # Generate questions
    outputs = qg_model.generate(input_ids=input_ids, max_length=64, num_beams=10, num_return_sequences=5)
    questions = [qg_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    qas = []
    for question in questions:
        inputs = qa_tokenizer.encode_plus(question, paragraph, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].tolist()[0]

        # Perform inference to get the answer
        with torch.no_grad():
            outputs = qa_model(**inputs)
        
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = qa_tokenizer.decode(input_ids[answer_start:answer_end], skip_special_tokens=True)

        qas.append({
            'question': question,
            'answer': answer
        })

    return qas
