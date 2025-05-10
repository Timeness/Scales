from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    if prompt:
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"response": response})
    else:
        return jsonify({"error": "No prompt provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)
