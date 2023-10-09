import keras
from segmentation_models.metrics import iou_score
from segmentation_models.losses import dice_loss
from flask import request, Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = "manbearpig_MUDMAN888"
model_file_name = "models/07_last_optim.h5"

@app.route("/", methods=['POST', 'GET'])
def affichage_result_mask(_img):
    
    _img = request.files['image']
    # load the model and detect
    loaded_model = keras.models.load_model(model_file_name ,
                                    custom_objects={
                                        "dice_loss": dice_loss,
                                        "iou_score": iou_score
                                        })

    mask_u = loaded_model.predict(_img)

    return mask_u
