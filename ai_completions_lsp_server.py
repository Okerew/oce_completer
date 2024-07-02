from flask import Flask
from flask_socketio import SocketIO, emit
from transformers import AutoTokenizer, AutoModelForCausalLM
import ast

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('completions')
def handle_completions(data):
    text = data.get('text', '')
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Generate a longer sequence
    output = model.generate(input_ids, max_length=input_ids.shape[1] + 50, num_return_sequences=1, do_sample=True,
                            top_k=50, top_p=0.95)

    # Decode the generated sequence
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the input text from the generated text
    completion = generated_text[len(text):]

    # Split the completion into words
    words = completion.split()

    # Create suggestions based on the first 5 words or fewer if there aren't 5
    suggestions = []
    for i in range(min(5, len(words))):
        suggestion = ' '.join(words[:i + 1])
        suggestions.append(text + suggestion)

    return suggestions


@socketio.on('lint')
def handle_lint(data):
    text = data.get('text', '')
    errors = []
    try:
        ast.parse(text)
    except SyntaxError as e:
        errors.append({
            'line': e.lineno,
            'column': e.offset,
            'message': str(e)
        })
    return errors


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=3000, allow_unsafe_werkzeug=True)