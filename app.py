from flask import Flask, jsonify, request
from service import paraphrases

app = Flask(__name__)


@app.route('/paraphrase')
def paraphrase():
    input_str = request.args.get('tree').replace('%', ' ')

    output = paraphrases(input_str)

    return jsonify(output)


if __name__ == '__main__':
    app.run()
