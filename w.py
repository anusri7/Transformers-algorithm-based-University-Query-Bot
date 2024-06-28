from fastapi import FastAPI, Query, Request
import requests
import json
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import torch.optim as optim


app = FastAPI()

@app.get("/webhook")
async def webhook(
    hub_mode: str = Query(..., alias="hub.mode"),
    hub_challenge: str = Query(..., alias="hub.challenge"),
    hub_verify_token: str = Query(..., alias="hub.verify_token")
):
    # Process the received parameters
    print({"hub_mode": hub_mode, "hub_challenge": hub_challenge, "hub_verify_token": hub_verify_token})
    return int(hub_challenge)


def pretrain_beforestart():
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
            
        }, 
        "20tuit002": {
        "cgpa": 10, 
        "overall percentage": 99,
        "5thsemgpa": 6.52,
        "2ndsemgpa": 7.3,
        "fees status": "paid",
        "overall performance": "good",
        "attendance percentage": 75,
        "internal marks": 92,
        "training attendance": 70,
        
    }
    }

    # # Sample questions
    # questions = [
    #     "What is the CGPA of 20tuit001?",
    #     "What is the overall percentage of 20tuit001?",
    #     "What is the GPA of 20tuit001 in the 5th semester?",
    #     "What is the GPA of 20tuit001 in the 2nd semester?",
    #     "Is the fees status of 20tuit001 paid?",
    #     "How would you describe the overall performance of 20tuit001?",
    #     "What is the attendance percentage of 20tuit001?",
    #     "What are the internal marks of 20tuit001?",
    #     "What is the training attendance percentage of 20tuit001?",
    #     "Is 20tuit001's overall percentage above 80?"
    # ]

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
    for epoch in range(10):  # Adjust number of epochs as needed
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

    

    return model
@app.post("/train_model")
async def train_model(request: Request):
    pretrain_beforestart()
    # Process the received payload
    
    
    # tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    # model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    # payload = await request.json()
    # print(payload)
    # # Define optimizer and learning rate
    # optimizer = optim.AdamW(model.parameters(), lr=5e-5)  # AdamW is recommended for transformers models

    # # Prepare the training data
    # training_data = []
    # for item in payload['payload']:
    #     for pattern, response in zip(item["patterns"], item["responses"]):
    #         input_text = pattern
    #         target_text = response
    #         training_data.append((input_text, target_text))
    # print("heree")
    # # Fine-tune the model
    # model.train()
    # result = ""
    # print("heree iterate")

    # for epoch in range(1):  # Adjust number of epochs as needed
    #     total_loss = 0
    #     for input_text, target_text in training_data:
    #         optimizer.zero_grad()
    #         input_ids = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).input_ids
    #         target_ids = tokenizer(target_text, return_tensors="pt", max_length=1024, truncation=True).input_ids
    #         loss = model(input_ids=input_ids, labels=target_ids).loss
    #         total_loss += loss.item()
    #         loss.backward()
    #         optimizer.step()
    #     result += f"Epoch {epoch + 1}, Loss: {total_loss / len(training_data)} \n" 
    # model.save_pretrained("model.pt")
    # tokenizer.save_pretrained("token.pt")

    return "success"

# Define a function to generate a response
def generate_response(user_input):
    # Define some predefined responses
    # Tokenize the input text
    # Test the chatbot
    model = BartForConditionalGeneration.from_pretrained("model.pt")
    tokenizer = BartTokenizer.from_pretrained("token.pt")
    response = "Sorry i cant understand you."
    model.eval()
    with torch.no_grad():
        for question in [user_input]:
            input_ids = tokenizer(question, return_tensors="pt", max_length=1024, truncation=True).input_ids
            generated_ids = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
            response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Question: {question}\nAnswer: {response}\n")

    return  response

def send_trigger_to_app(payload, res):

    print(res)
    url = "https://graph.facebook.com/v18.0/189324020936186/messages"
    phone_number = payload['contacts'][0]['wa_id']
    payload = json.dumps({
    "messaging_product": "whatsapp",
    "to": phone_number,
    "type": "template",
    "template": {
        "name": "college_reply",
        "language": {
        "code": "en_US"
        },
         "components": [
      {
        "type": "body",
         "parameters": [{
            "type": "text",
            "text":res
        }]
      }
    ]
  },
  }
  )
    headers = {
    'Authorization': 'Bearer EAATaL5pTKZAYBO5EU9VsWSg0JfvGOAbRpqPzNr6iIwBOYnuI6vHZBPEBi57GqqUjRJzxsaXpODdlfXUcmjdN4LapcyGAZAyes0HCu1WSyTopuSZCCmGgtxOm5n6sWIc44cngnqBqDwCCUwqHnFwHUnXzxbp7TE1y3dcnwPMNdW7ZByVUqIN9R1c3omVKrwmugX93pthzILZCmPKfXbyDcZD',
    'Content-Type': 'application/json',
    'Cookie': 'ps_l=0; ps_n=0'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)



@app.post("/webhook")
async def webhook(request: Request):
    # Process the received payload
    payload = await request.json()
    print(payload)
    # {'object': 'whatsapp_business_account', 'entry': [{'id': '152085297998149', 'changes': [{'value': {'messaging_product': 'whatsapp', 'metadata': {'display_phone_number': '15550271697', 'phone_number_id': '189324020936186'}, 'contacts': [{'profile': {'name': 'Veejay Vivu'}, 'wa_id': '919943814594'}], 'messages': [{'from': '919943814594', 'id': 'wamid.HBgMOTE5OTQzODE0NTk0FQIAEhgWM0VCMEU4ODFENDFCNjg5MDMyQjUzRAA=', 'timestamp': '1709306214', 'text': {'body': 'percentage'}, 'type': 'text'}]}, 'field': 'messages'}]}]}
    res = {}
    if  payload['entry'][0]['changes'][0]['value'].get("messages") and payload['entry'][0]['changes'][0]['value'].get("messages")[0].get("text"):
        input_x  = payload['entry'][0]['changes'][0]['value']["messages"][0]['text']['body']
        res = generate_response(input_x) 
        send_trigger_to_app( payload['entry'][0]['changes'][0]['value'] ,res)
    # You can now access the payload as a dictionary
    print(res)
    return res