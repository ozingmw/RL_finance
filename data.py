import json

with open('005930.json', 'r') as f:
    df = json.load(f)

    
def refine(data):
    refine_data = {
        "price": data['stck_prpr']
    }