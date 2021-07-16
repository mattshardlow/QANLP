from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch

from flask import render_template, request, Flask

tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
model = BertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny')
optim = AdamW(model.parameters(), lr=5e-5)

app = Flask(__name__)


corpus = []
counter = 0
neg = 0
question = ""
loss = ""
decision = []

@app.route('/', methods=['GET','POST'])
def home():
    global corpus
    global counter
    global neg
    global decision

    corpus = []
    counter = 0
    neg = 0
    decision = []
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    global neg
    global counter
    global corpus
    global question
    global loss

    if counter == 0:
        question   = request.form['question']
        corpus_str = request.form['corpus']
        corpus     = corpus_str.split("\n")

    if counter != len(corpus):
        if len(decision) < counter:
          decision.append("Yes")
        text = corpus[counter]
        counter = counter + 1
    else:
        return complete()

    qu_text = question + " [SEP] " + text

    inputs = tokenizer(qu_text, return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)
    outputs = model(**inputs, labels=labels)

    logits = outputs.logits
    loss = outputs.loss

    pred = "Yes" if logits[0][0] < logits[0][1] else "No"
    print(logits)

    correct = counter-neg-1
    remain = len(corpus)-counter
   

    res = render_template('process.html', prediction=pred, text=text, question=question, count=counter-1, correct=correct, remaining = remain)
    return res

@app.route('/process_no', methods=['POST'])
def process_no():
    global neg
    global loss
    neg = neg + 1
    loss.backward()
    optim.step()
    decision.append("No")

    return process()
    

@app.route('/complete', methods=['POST'])
def complete():
    global counter
    global question
    table= "<table style=\"width:100\%\"><tr><th>Text</th><th>Machine Prediction</th><th>User Decision</th></tr>"
    for i, entry in enumerate(corpus):
        qu_text = question + " [SEP] " + entry
        inputs = tokenizer(qu_text, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits
        pred = "Yes" if logits[0][0] < logits[0][1] else "No"
        this_decision = "-"
        if i < len(decision):
            this_decision = decision[i]
        table += "<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (entry, pred, this_decision)
    table += "</table>"

    return render_template('complete.html', num=counter, pred_table=table)
