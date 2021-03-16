import json
import os

domains = {
  'indomain': ['nat_questions', 'newsqa', 'squad'],
  'oodomain': ['duorc', 'race', 'relation_extraction']
}

classifications = ['train', 'val']

def main(): 
  for inOrOut in domains:
    for dataset in domains[inOrOut]:
      for classification in classifications:
        input_path = 'datasets/' + inOrOut + '_' + classification + '/' + dataset
        extract(input_path)
  print("Finished")
    

def extract(input_path):
  output_path = "augmentation/" + input_path + "-questions"
  if not os.path.exists(os.path.dirname(output_path)):
    try:
        os.makedirs(os.path.dirname(output_path))
    except:
      print ("Failed extracting questions")

  output_file = open(output_path, 'w')

  with open(input_path, "r") as input_file:
    text = input_file.read()
    text_dict = json.loads(text)
    for dataDict in text_dict['data']:
      title = dataDict['title']
      paragraphs = dataDict['paragraphs']
      for paragraph in paragraphs:
        context = paragraph['context']
        qas = paragraph['qas']
        for qa in qas:
          question = qa['question'] + '\n'
          answer = qa['answers']
          output_file.write(question)
  output_file.close()
  input_file.close()


if __name__ == "__main__":
  print("This script is used to extract the questions from dataset text files.")
  main()