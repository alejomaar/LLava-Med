import pandas as pd
import json
from PIL import Image
import io

#df_test =  pd.read_parquet('https://huggingface.co/datasets/flaviagiammarino/vqa-rad/resolve/main/data/test-00000-of-00001-e5bc3d208bb4deeb.parquet')
#image_bytes = df_test.head(1).to_dict(orient='records')[0]['image']['bytes']
#image = Image.open(io.BytesIO(image_bytes))
#image.save('output_image.png')

def prepare(info):
    #print(info)
    return {'id': info['qid'],
    'image': info['image_name'],
    'answer_type': info['answer_type'],
    'conversations':
      [{'from': 'human', 'value': str(info['question']).lower()},
       {'from': 'gpt', 'value': str(info['answer']).lower()}
    ]}

with open('VQA-RAD/VQA_RAD Dataset Public.json', 'r') as file:
    data = json.load(file)

llava_format = [prepare(row) for row in data]

with open('data/test.json', 'w', encoding='utf-8') as f:
    json.dump(llava_format, f, ensure_ascii=False, indent=4)
