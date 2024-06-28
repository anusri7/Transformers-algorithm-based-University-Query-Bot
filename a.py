from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import torch.optim as optim

# Define the data for fine-tuning
data = {
    "20tuit001": {
        "cgpa": 8.24, 
        "overall percentage": 89,
        "5thsemgpa": 8.52,
        "2ndsemgpa": 8.3,
        "fees status": "paid",
        "overall performance": "good",
        "attendance percentage": 85,
        "internal marks": 90,
        "training attendance": 80,
        
    }
    
}

# Sample questions
questions = [
    #   "What is the CGPA of 20tuit001?",
    # "What is the overall percentage of 20tuit001?",
    #  "What is the GPA of 20tuit00 in the 5th semester?",
    #  "What is the GPA of 20tuit001 in the 2nd semester?",
    # "Is the fees status of 20tuit001 paid?",
    #  "How would you describe the overall performance of 20tuit001?",
    # "What is the attendance percentage of 20tuit001?",
    #  "What are the internal marks of 20tuit001?",
    #  "What is the training attendance percentage of 20tuit001?",
    #    "Is 20tuit001's overall percentage above 80?"
]

# Initialize the BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Define optimizer and learning rate
optimizer = optim.AdamW(model.parameters(), lr=5e-5)  # AdamW is recommended for transformers models

# Prepare the training data
training_data = []
for name, info in data.items():
    input_text_gpa = f"What is the CGPA of {name}?"
    target_text_gpa = f"{name}'s CGPA is {info['cgpa']}."
    training_data.append((input_text_gpa, target_text_gpa))

    input_text_percentage = f"What is the percentage of {name}?"
    target_text_percentage = f"{name}'s percentage is {info['overall percentage']}."
    training_data.append((input_text_percentage, target_text_percentage))


    # Additional questions
    input_text_2nd_sem_gpa = f"What is the GPA of {name} in the 2nd semester?"
    target_text_2nd_sem_gpa = f"{name}'s GPA in the 2nd semester is {info['2ndsemgpa']}."
    training_data.append((input_text_2nd_sem_gpa, target_text_2nd_sem_gpa))

    input_text_5th_sem_gpa = f"What is the GPA of {name} in the 5th semester?"
    target_text_5th_sem_gpa = f"{name}'s GPA in the 5th semester is {info['5thsemgpa']}."
    training_data.append((input_text_5th_sem_gpa, target_text_5th_sem_gpa))

    input_text_fees_status = f"Is the fees status of {name} paid?"
    target_text_fees_status = f"Yes, the fees status of {name} is paid." if info['fees status'] == 'paid' else f"No, the fees status of {name} is not paid."
    training_data.append((input_text_fees_status, target_text_fees_status))

    input_text_performance = f"How would you describe the overall performance of {name}?"
    target_text_performance = f"The overall performance of {name} is {info['overall performance']}."
    training_data.append((input_text_performance, target_text_performance))

    input_text_attendance = f"What is the attendance percentage of {name}?"
    target_text_attendance = f"{name}'s attendance percentage is {info['attendance percentage']}%."
    training_data.append((input_text_attendance, target_text_attendance))

    input_text_internal_marks = f"What are the internal marks of {name}?"
    target_text_internal_marks = f"{name}'s internal marks are {info['internal marks']}."
    training_data.append((input_text_internal_marks, target_text_internal_marks))

    input_text_training_attendance = f"What is the training attendance percentage of {name}?"
    target_text_training_attendance = f"{name}'s training attendance percentage is {info['training attendance']}%."
    training_data.append((input_text_training_attendance, target_text_training_attendance))

    

    input_text_above_80_percentage = f"Is {name}'s overall percentage above 80?"
    target_text_above_80_percentage = f"Yes, {name}'s overall percentage is above 80." if info['overall percentage'] > 80 else f"No, {name}'s overall percentage is not above 80."
    training_data.append((input_text_above_80_percentage, target_text_above_80_percentage))

# Fine-tune the model
model.train()
for epoch in range(5):  # Adjust number of epochs as needed
    total_loss = 0
    for input_text, target_text in training_data:
        optimizer.zero_grad()
        input_ids = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).input_ids
        target_ids = tokenizer(target_text, return_tensors="pt", max_length=1024, truncation=True).input_ids
        loss = model(input_ids=input_ids, labels=target_ids).loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(training_data)}")

# Test the fine-tuned chatbot
model.eval()
with torch.no_grad():
    for question in questions:
        input_ids = tokenizer(question, return_tensors="pt", max_length=1024, truncation=True).input_ids
        generated_ids = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Question: {question}\nAnswer: {response}\n")
