from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import json

from tools.tools import convertToBuffer
from visualize.visualize import removeBgFromSegmentImage, removeOnlyBg
from models.model import getMask, loadModel
from models.preprocess import preprocess

app = Flask(__name__)
CORS(app)

FAST_SAM = loadModel()


@app.route('/segment', methods=['POST'])
def segment_marker():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    img_rgb = Image.open(image).convert('RGB')

    marker_coordinates = request.form.get('markers-coordinates', None)
    if not marker_coordinates:
        return jsonify({'error': 'No markers provided'}), 400

    marker_coordinates = json.loads(marker_coordinates)

    try:

        input_points, input_labels = preprocess(marker_coordinates)

        masks = getMask(img_rgb, FAST_SAM, input_points, input_labels)

        bg_removed_segmented_img = removeBgFromSegmentImage(img_rgb, masks[0])
        img_base64_bg_segmented = convertToBuffer(bg_removed_segmented_img)

        bg_only_removed_img = removeOnlyBg(img_rgb, masks[0])
        img_base64_only_bg = convertToBuffer(bg_only_removed_img)

        return jsonify({'bg_removed_segmented_img': f'data:image/png;base64,{img_base64_bg_segmented}',
                        'bg_only_removed_segmented_img': f'data:image/png;base64,{img_base64_only_bg}'}), 200

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the image'}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
