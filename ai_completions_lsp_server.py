from flask import Flask
from flask_socketio import SocketIO
from transformers import AutoTokenizer, AutoModelForCausalLM
import ast
from collections import defaultdict

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

pad_token_id = tokenizer.eos_token_id  # Set pad token ID to end-of-sequence token ID

# Initialize a dictionary to cache patterns
pattern_cache = defaultdict(list)


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('completions')
def handle_completions(data):
    text = data.get('text', '')

    # Check if we have cached patterns for the given text
    if text in pattern_cache:
        suggestions = pattern_cache[text]
    else:
        input_ids = tokenizer.encode(text, return_tensors='pt')

        # Create an attention mask
        attention_mask = input_ids.ne(pad_token_id).long()

        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + 50,
            num_return_sequences=1,
            do_sample=True,
            top_k=20,
            top_p=0.80,
            pad_token_id=pad_token_id,
            temperature=0.5,
            repetition_penalty=1.2,
            length_penalty=1.0
        )

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

        # Cache the generated patterns
        pattern_cache[text] = suggestions

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
    socketio.run(app, host='0.0.0.0', port=3000, allow_unsafe_werkzeug=True, log_output=True, use_reloader=False)
