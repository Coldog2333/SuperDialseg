from flask import Flask, request, render_template
from super_dialseg import (
    TexttilingSegmenter,
    ChatGPTSegmenter,
    EvenSegmenter,
    TexttilingNSPSegmenter,
    TextsegSegmenter
)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def homepage():
    segmented_dialogue = None
    if request.method == 'POST':
        dialogue = request.form['dialogue']
        model_id = request.form['model_id']
        segmented_dialogue = dialogue_segmentation(dialogue, model_id)
        segmented_dialogue = segmented_dialogue.replace('\n', '<br/>')
    return render_template('index.html', segmented_dialogue=segmented_dialogue)


# Return segmented dialogue
def dialogue_segmentation(dialogue: str, model_id: str) -> str:
    utterances = dialogue.split('\n')
    if model_id == 'TextTilling':
        segmenter = TexttilingSegmenter()
    elif model_id == 'TextTilling+NSP':
        segmenter = TexttilingNSPSegmenter()
    elif model_id == 'TextSeg-super_dialseg':
        segmenter = TextsegSegmenter(
            model_name_or_path='../.cache/model_zoo/textseg/textseg_super_dialseg.pkl',
            word2vecfile='../.cache/model_zoo/word2vec/GoogleNews-vectors-negative300-10w.bin',
            threshold=0.4,
            device='cpu'
        )
    elif model_id == 'ChatGPT':
        from secret_config import OPENAI_KEY
        segmenter = ChatGPTSegmenter(openai_key=OPENAI_KEY)
    else:  # even
        segmenter = EvenSegmenter()

    _, segmented_dialogue = segmenter(utterances, auto_print=False)
    return segmented_dialogue


if __name__ == '__main__':
    app.run(debug=True)
